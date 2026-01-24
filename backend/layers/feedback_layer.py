"""
Feedback Layer

Input: Code context from VS Code
Action: Generate feedback (on request)
Output: Structured feedback items ready for rendering in VS Code

This layer is a feedback generation service. It does not decide when 
feedback should be shown. Instead, it generates candidate feedback items 
when requested by the Control layer. Feedback generation may use an LLM 
and can be cached to reduce latency.

Output is structured to support editor rendering and highlighting.
"""

import asyncio
import json
import time
import hashlib
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import uuid

from backend.types import (
    CodeContext,
    FeedbackItem,
    FeedbackResponse,
    FeedbackMetadata,
    FeedbackType,
    FeedbackPriority,
    UserStateEstimate,
)
from backend.types.code_context import CodePosition, CodeRange
from backend.types.config import FeedbackLayerConfig
from backend.layers.llm_client import LLMClient, create_llm_client, LLMResponse


@dataclass
class _CacheEntry:
    response: FeedbackResponse
    created_at: float  # unix seconds


class FeedbackLayer:
    def __init__(self, config: Optional[FeedbackLayerConfig] = None):
        self._config = config or FeedbackLayerConfig()

        # TTL cache
        self._cache: Dict[str, _CacheEntry] = {}

        # Rate limiting: store timestamps of recent generations
        self._rate_limit_window: List[float] = []

        # Optional: prevent duplicate concurrent LLM calls for same key
        self._inflight: Dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()

        self._llm_client: Optional[LLMClient] = None
        self._min_priority: Optional[FeedbackPriority] = None

        # Simple stats
        self._cache_hits = 0
        self._cache_misses = 0

    def configure(self, config: FeedbackLayerConfig) -> None:
        self._config = config

    def initialize_llm(self) -> bool:
        """
        Initialize the LLM client based on configuration.
        """
        print("[FeedbackLayer] Initializing LLM client")
        self.set_llm_client(create_llm_client(
            provider=self._config.llm_provider,
            api_key=self._config.llm_api_key,
            model=self._config.llm_model,
        ))
        
        configured = self._llm_client is not None and self._llm_client.is_configured()
        if configured:
            print(f"[FeedbackLayer] LLM initialized: {self._llm_client.get_model_name()}")
        else:
            print("[FeedbackLayer] LLM not configured")
        
        return configured
    
    def set_llm_client(self, client: LLMClient) -> None:
        """
        Inject a custom LLM client (useful for testing).
        """
        self._llm_client = client

    def shutdown_llm(self) -> None:
        self._llm_client = None

    async def generate_feedback(
        self,
        context: CodeContext,
        user_state: Optional[UserStateEstimate] = None,
        feedback_types: Optional[List[FeedbackType]] = None,
    ) -> FeedbackResponse:
        """
        The non-cached core. This should be:
        - rate-limited
        - resilient (fallback on errors)
        - deterministic output shape
        """
        print("[FeedbackLayer] Generating feedback")
        start = time.time()

        if self.is_rate_limited():
            items = self._generate_fallback_feedback(context)
            items = self._filter_and_rank_feedback(items)
            return FeedbackResponse(
                items=items,
                request_id=self._make_request_id(context),
                total_generation_time_ms=(time.time() - start) * 1000.0,
                success=True,
                metadata=self._create_feedback_metadata(
                    cached=False,
                    generation_time_ms=(time.time() - start) * 1000.0,
                ),
            )

        self._update_rate_limit_window()

        try:
            # LLM path if configured
            if self._llm_client and self._llm_client.is_configured():
                print(f"[FeedbackLayer] Using LLM for feedback generation ({self._llm_client.get_model_name()})")
                prompt = self._build_llm_prompt(context, user_state)
                raw = await self._call_llm(prompt)
                items = self._parse_llm_response(raw, context)
            else:
                print("[FeedbackLayer] LLM not configured, using fallback")
                items = self._generate_fallback_feedback(context)

            items = self._filter_and_rank_feedback(items)

            return FeedbackResponse(
                items=items,
                request_id=self._make_request_id(context),
                total_generation_time_ms=(time.time() - start) * 1000.0,
                success=True,
                metadata=self._create_feedback_metadata(
                    cached=False,
                    generation_time_ms=(time.time() - start) * 1000.0,
                    generated_at=time.time(),
                    cache_key=self._compute_cache_key(context),
                    feedback_id=str(uuid.uuid4()),
                    session_id=self._get_session_id(context),
                    extra={
                        "used_llm": self._llm_client.get_model_name() if self._llm_client else "none",
                    }
                ),
            )
        except Exception as e:
            print(f"[FeedbackLayer] Error during feedback generation, using fallback: {e}")
            # Never fail hard in editor loop
            items = self._generate_fallback_feedback(context)
            items = self._filter_and_rank_feedback(items)
            return FeedbackResponse(
                items=items,
                request_id=self._make_request_id(context),
                total_generation_time_ms=(time.time() - start) * 1000.0,
                success=False,
                metadata=self._create_feedback_metadata(
                    cached=False,
                    generation_time_ms=(time.time() - start) * 1000.0,
                    generated_at=time.time(),
                    cache_key="",
                    feedback_id=str(uuid.uuid4()),
                    session_id=self._get_session_id(context),
                    extra={
                        "error": str(e)
                    }
                ),
            )

    async def generate_feedback_cached(
        self,
        context: CodeContext,
        user_state: Optional[UserStateEstimate] = None,
        feedback_types: Optional[List[FeedbackType]] = None,
    ) -> FeedbackResponse:
        """
        Caching + optional single-flight.
        """
        print("[FeedbackLayer] Generating cached feedback")
        cache_key = self._compute_cache_key(context)

        if not self._config.enable_cache:
            print("[FeedbackLayer] Cache disabled, generating fresh feedback")
            return await self.generate_feedback(context, user_state, feedback_types)

        # Fast path: try cache
        cached = self._check_cache(cache_key)
        if cached is not None:
            return cached

        # Optional single-flight to avoid stampede
        async with self._lock:
            cached2 = self._check_cache(cache_key)
            if cached2 is not None:
                return cached2

            if cache_key in self._inflight:
                fut = self._inflight[cache_key]
            else:
                fut = asyncio.get_event_loop().create_future()
                self._inflight[cache_key] = fut

        if fut.done():
            return fut.result()

        # Only the creator should compute; others await
        creator = False
        async with self._lock:
            creator = (self._inflight.get(cache_key) is fut) and (not fut.done())

        if creator:
            try:
                resp = await self.generate_feedback(context, user_state, feedback_types)
                self._store_in_cache(cache_key, resp)
                fut.set_result(resp)
                return resp
            except Exception as e:
                fut.set_exception(e)
                raise
            finally:
                async with self._lock:
                    self._inflight.pop(cache_key, None)
        else:
            return await fut

    def invalidate_cache(self, file_path: Optional[str] = None) -> None:
        """
        Invalidate cache entries.

        AI-generated code - should be checked further later
        
        :param self: Object instance
        :param file_path: Optional file path to invalidate; if None, clear entire cache
        :type file_path: Optional[str]
        """
        if file_path is None:
            self._cache.clear()
            return

        # naive prefix match because key includes file_path in hashed payload,
        # so we keep an auxiliary strategy: remove entries that mention file_path
        # by recomputing stored "request_id" isn't ideal. Simpler: store file_path
        # in cache entry later if you want exact invalidation.
        to_remove = []
        for k, entry in self._cache.items():
            # best-effort: if you include file_path in the unhashed payload below,
            # you can also store a side-map.
            if file_path in k:
                to_remove.append(k)
        for k in to_remove:
            self._cache.pop(k, None)

    def get_cache_stats(self) -> Dict[str, Any]:
        # Prune expired first to keep stats meaningful
        self._prune_expired_cache()
        return {
            "enabled": self._config.enable_cache,
            "size": len(self._cache),
            "ttl_seconds": self._config.cache_ttl_seconds,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
        }

    def is_rate_limited(self) -> bool:
        now = time.time()
        window_seconds = 60.0
        cutoff = now - window_seconds
        self._rate_limit_window = [t for t in self._rate_limit_window if t >= cutoff]
        limited = len(self._rate_limit_window) >= self._config.max_generations_per_minute
        if limited:
            print("[FeedbackLayer] Rate limited")
        return limited

    def set_feedback_priority_filter(self, min_priority: FeedbackPriority) -> None:
        self._min_priority = min_priority

    # --- Internal methods ---

    def _make_request_id(self, context: CodeContext) -> str:
        # stable-ish request id for logs; not same as cache key
        payload = f"{context.file_path}:{getattr(context, 'timestamp', 0.0)}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]

    def _get_session_id(self, context: CodeContext) -> str:
        """
        Docstring for _get_session_id
        
        :param self: Object instance
        :param context: Code context
        :type context: CodeContext
        :return: Session ID string
        :rtype: str
        """

        return "TODO"

    def _compute_cache_key(self, context: CodeContext) -> str:
        """
        Use a hash of relevant context, not just file name.

        AI-generated code - should be optimized further later

        """
        cursor = getattr(context, "cursor_position", None)
        selection = getattr(context, "selection", None)
        diagnostics = getattr(context, "diagnostics", []) or []

        # Prefer visible window if you have it; otherwise take a small slice around cursor
        content = context.file_content or ""
        if content and cursor is not None:
            # small, stable slice: +/- N lines around cursor
            lines = content.splitlines()
            line_idx = max(0, min(cursor.line, len(lines) - 1))
            a = max(0, line_idx - 40)
            b = min(len(lines), line_idx + 40)
            focus_text = "\n".join(lines[a:b])
        else:
            focus_text = content[:4000]  # cap

        # Compact diagnostics: cap count and message length
        diag_compact: List[Tuple[str, int, int]] = []
        for d in diagnostics[:10]:
            msg = getattr(d, "message", "")[:120]
            sev = getattr(d, "severity", "")
            rng = getattr(d, "range", None)
            # best-effort range extraction
            if rng and getattr(rng, "start", None):
                ln = rng.start.line
                ch = rng.start.character
            else:
                ln, ch = -1, -1
            diag_compact.append((f"{sev}:{msg}", ln, ch))

        key_obj = {
            "file_path": context.file_path,
            "language_id": context.language_id,
            "cursor": None if cursor is None else {"line": cursor.line, "character": cursor.character},
            "selection": None if selection is None else {
                "start": {"line": selection.start.line, "character": selection.start.character},
                "end": {"line": selection.end.line, "character": selection.end.character},
            },
            "focus_text": focus_text,
            "diagnostics": diag_compact,
        }

        raw = json.dumps(key_obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _check_cache(self, cache_key: str) -> Optional[FeedbackResponse]:
        self._prune_expired_cache()
        entry = self._cache.get(cache_key)
        if not entry:
            self._cache_misses += 1
            print("[FeedbackLayer] Cache miss")
            return None

        self._cache_hits += 1
        print("[FeedbackLayer] Cache hit")
        resp = entry.response

        # Mark metadata as cached (if you have that field)
        try:
            if getattr(resp, "metadata", None):
                resp.metadata.cached = True
        except Exception:
            pass

        return resp

    def _store_in_cache(self, cache_key: str, response: FeedbackResponse) -> None:
        self._cache[cache_key] = _CacheEntry(response=response, created_at=time.time())

    def _prune_expired_cache(self) -> None:
        if not self._config.enable_cache:
            self._cache.clear()
            return

        ttl = float(self._config.cache_ttl_seconds)
        if ttl <= 0:
            self._cache.clear()
            return

        cutoff = time.time() - ttl
        expired = [k for k, v in self._cache.items() if v.created_at < cutoff]
        for k in expired:
            self._cache.pop(k, None)

    from typing import Optional

    def _build_llm_prompt(
        self,
        context: CodeContext,
        user_state: Optional["UserStateEstimate"] = None,
    ) -> str:
        cursor = context.cursor_position

        def _fmt_pos(p: CodePosition) -> str:
            return f"line {p.line}, col {p.character}"

        def _fmt_range(r: CodeRange) -> str:
            return f"{_fmt_pos(r.start)} -> {_fmt_pos(r.end)}"

        # Diagnostics text (aligned with DiagnosticInfo / DiagnosticSeverity / CodeRange)
        diag_lines = []
        for d in (context.diagnostics or []):
            sev = d.severity.value if hasattr(d.severity, "value") else str(d.severity)
            src = d.source or "n/a"
            code = d.code or "n/a"
            diag_lines.append(
                f"- [{sev}] {d.message} @ {_fmt_range(d.range)} (source={src}, code={code})"
            )
        diag_text = "\n".join(diag_lines) if diag_lines else "(none)"

        selection_text = (
            f"{_fmt_range(context.selection)}"
            + (f"\nSelection content:\n{context.selection.content}"
            if context.selection and context.selection.content else "")
            if context.selection else "(none)"
        )

        visible_text = (
            f"{_fmt_range(context.visible_range)}"
            + (f"\nVisible content:\n{context.visible_range.content}"
            if context.visible_range and context.visible_range.content else "")
            if context.visible_range else "(none)"
        )

        # Code to show: prefer file_content; otherwise fall back to selection/visible content if present
        code = (
            context.file_content
            or (context.visible_range.content if context.visible_range and context.visible_range.content else None)
            or (context.selection.content if context.selection and context.selection.content else None)
            or ""
        )

        # Optional user_state snippet (kept generic so it doesn't break if UserStateEstimate changes)
        user_state_text = "(none)"
        if user_state is not None:
            # keep it robust: don't assume exact fields exist
            try:
                user_state_text = str(user_state)
            except Exception:
                user_state_text = "<unprintable user_state>"

        return (
            "You are an expert programming assistant. You will receive VS Code code context.\n"
            "Your task: generate concise, actionable feedback items focused on the most likely issue near the cursor.\n\n"

            "Return ONLY valid JSON (no markdown, no extra keys) with exactly this schema:\n"
            "{\n"
            '  "items": [\n'
            "    {\n"
            '      "title": string,\n'
            '      "message": string,\n'
            '      "type": "hint" | "suggestion" | "warning" | "explanation" | "simplification",\n'
            '      "priority": "low" | "medium" | "high"\n'
            '      "code_range": {\n'
            '        "start": {"line": int, "character": int},\n'
            '        "end": {"line": int, "character": int}\n'
            '      },\n'
            '      "confidence": float (0.0 to 1.0),\n'
            '      "dismissible": boolean,\n'
            '      "actionable": boolean,\n'
            '      "action_label": boolean,\n'
            "    }\n"
            "  ]\n"
            "}\n\n"

            "Constraints:\n"
            f"- max items: {self._config.max_feedback_items}\n"
            f"- max message length per item: {self._config.max_message_length}\n"
            "- Focus on the most likely issue near the cursor.\n"
            "- If diagnostics exist, prioritize them.\n"
            "- Do NOT invent errors not supported by code/diagnostics.\n"
            "- Prefer specific edits (what/where/how) over generic advice.\n\n"

            "Context:\n"
            f"- language_id: {context.language_id}\n"
            f"- file_path: {context.file_path}\n"
            f"- workspace_folder: {context.workspace_folder or '(none)'}\n"
            f"- timestamp: {context.timestamp}\n"
            f"- cursor_position: {_fmt_pos(cursor)}\n"
            f"- selection: {selection_text}\n"
            f"- visible_range: {visible_text}\n"
            f"- metadata: {context.metadata if context.metadata else '{}'}\n"
            f"- user_state: {user_state_text}\n\n"

            "Diagnostics (DiagnosticInfo.severity is one of: error, warning, info, hint):\n"
            f"{diag_text}\n\n"

            "Code (from context.file_content if available):\n"
            "```"
            f"{context.language_id}\n{code}\n"
            "```"
        )

    async def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM client to generate feedback.
        
        Returns the raw response content as a string.
        Raises RuntimeError if the client is not configured or call fails.
        """
        if not self._llm_client:
            raise RuntimeError("LLM client not initialized")
        
        if not self._llm_client.is_configured():
            raise RuntimeError("LLM client not configured")
        
        response: LLMResponse = await self._llm_client.generate(
            prompt=prompt,
            max_tokens=500,
            temperature=0.3,
        )
        
        if not response.success:
            raise RuntimeError(f"LLM call failed: {response.error}")
        
        print(f"[FeedbackLayer] LLM response received in {response.latency_ms:.0f}ms (tokens: {response.usage.get('total_tokens', 0)})")
        
        return response.content

    def _parse_llm_response(self, response: str, context: CodeContext) -> List[FeedbackItem]:
        """
        Parse strict JSON. If it fails, fallback gracefully.
        """
        try:
            obj = json.loads(response)
            raw_items = obj.get("items", [])[: self._config.max_feedback_items]
            items: List[FeedbackItem] = []

            # Map string values to enums
            type_map = {
                "hint": FeedbackType.HINT,
                "suggestion": FeedbackType.SUGGESTION,
                "warning": FeedbackType.WARNING,
                "explanation": FeedbackType.EXPLANATION,
                "simplification": FeedbackType.SIMPLIFICATION,
            }
            priority_map = {
                "low": FeedbackPriority.LOW,
                "medium": FeedbackPriority.MEDIUM,
                "high": FeedbackPriority.HIGH,
                "critical": FeedbackPriority.CRITICAL,
            }

            for it in raw_items:
                title = str(it.get("title", "Feedback"))[:80]
                msg = str(it.get("message", ""))[: self._config.max_message_length]

                # Parse type and priority
                feedback_type = type_map.get(
                    str(it.get("type", "hint")).lower(),
                    FeedbackType.HINT
                )
                priority = priority_map.get(
                    str(it.get("priority", "medium")).lower(),
                    FeedbackPriority.MEDIUM
                )

                # Parse code_range if present
                code_range: Optional[CodeRange] = None
                raw_range = it.get("code_range")
                if raw_range and isinstance(raw_range, dict):
                    try:
                        start = raw_range.get("start", {})
                        end = raw_range.get("end", {})
                        code_range = CodeRange(
                            start=CodePosition(
                                line=int(start.get("line", 0)),
                                character=int(start.get("character", 0)),
                            ),
                            end=CodePosition(
                                line=int(end.get("line", 0)),
                                character=int(end.get("character", 0)),
                            ),
                        )
                    except (TypeError, ValueError):
                        code_range = None

                # Parse confidence (0.0 to 1.0)
                try:
                    confidence = float(it.get("confidence", 0.5))
                    confidence = max(0.0, min(1.0, confidence))
                except (TypeError, ValueError):
                    confidence = 0.5

                # Parse boolean fields
                dismissible = bool(it.get("dismissible", True))
                actionable = bool(it.get("actionable", False))

                # Parse action_label (string or None)
                action_label = it.get("action_label")
                if action_label is not None:
                    action_label = str(action_label)[:50] if action_label else None

                items.append(FeedbackItem(
                    title=title,
                    message=msg,
                    feedback_type=feedback_type,
                    priority=priority,
                    code_range=code_range,
                    confidence=confidence,
                    dismissible=dismissible,
                    actionable=actionable,
                    action_label=action_label,
                    metadata=self._create_feedback_metadata(
                        generated_at=time.time(),
                        generation_time_ms=0.0,
                        cache_key=self._compute_cache_key(context),
                        feedback_id=str(uuid.uuid4()), # TODO: generate ID that links to the response
                        session_id=self._get_session_id(context),
                        extra={
                            "llm_model": self._llm_client.get_model_name() if self._llm_client else "none",
                        },
                    ),
                ))

            return items or self._generate_fallback_feedback(context)
        except Exception as e:
            print(f"[FeedbackLayer] Failed to parse LLM response: {e}")
            return self._generate_fallback_feedback(context)

    def _generate_fallback_feedback(self, context: CodeContext) -> List[FeedbackItem]:
        """
        Cheap heuristics that always return something useful.
        """
        print("[FeedbackLayer] Generating fallback feedback")
        diags = context.diagnostics or []
        if diags:
            d0 = diags[0]
            msg = getattr(d0, "message", "There is a diagnostic message.")
            sev = getattr(d0, "severity", "unknown")
            return [
                FeedbackItem(
                    title=f"Diagnostic ({sev})",
                    message=str(msg)[: self._config.max_message_length],
                )
            ]

        # If no diagnostics, suggest “next action” near cursor
        return [
            FeedbackItem(
                title="Next step",
                message="Describe what you expect this code to do, then add a small print/log or a test at the cursor location to verify assumptions.",
            )
        ]

    def _create_feedback_metadata(
            self, 
            cached: bool = False, 
            generation_time_ms: float = 0.0, 
            generated_at: float = 0.0, 
            cache_key: str = "", 
            feedback_id: str = "", 
            session_id: str = "", 
            extra: dict = None
        ) -> FeedbackMetadata:
        """
        Adapt to your actual FeedbackMetadata fields.
        """
        print("[FeedbackLayer] Creating feedback metadata")
        return FeedbackMetadata(
            generated_at=generated_at or time.time(), 
            generation_time_ms=generation_time_ms,
            model_used=self._llm_client.get_model_name() if self._llm_client else None,
            cached=cached,
            cache_key=cache_key,
            feedback_id=feedback_id,
            session_id=session_id,
            extra=extra or {},
        )

    def _filter_and_rank_feedback(self, items: List[FeedbackItem]) -> List[FeedbackItem]:
        """
        Simple filtering and ranking:
        - enforce max items
        - priority by priority (if set) and confidence
        """
        items = items[: self._config.max_feedback_items]

        if self._min_priority is not None:
            items = [it for it in items if it.priority >= self._min_priority]
        items.sort(key=lambda it: (it.priority.value, it.confidence), reverse=True)

        return items

    def _update_rate_limit_window(self) -> None:
        now = time.time()
        self._rate_limit_window.append(now)
        # prune
        cutoff = now - 60.0
        self._rate_limit_window = [t for t in self._rate_limit_window if t >= cutoff]