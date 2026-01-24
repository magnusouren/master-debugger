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

from backend.types import (
    CodeContext,
    FeedbackItem,
    FeedbackResponse,
    FeedbackMetadata,
    FeedbackType,
    FeedbackPriority,
    UserStateEstimate,
)
from backend.types.config import FeedbackLayerConfig


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

        self._llm_client: Optional[object] = None  # replace later
        self._min_priority: Optional[FeedbackPriority] = None

        # Simple stats
        self._cache_hits = 0
        self._cache_misses = 0

    def configure(self, config: FeedbackLayerConfig) -> None:
        self._config = config

    def initialize_llm(self) -> bool:
        """
        Keep it simple for PoC: only validate config here.
        You can instantiate an actual client later.
        """
        if not self._config.llm_provider:
            self._llm_client = None
            return False

        # Example “configured” state; replace with real client later.
        if self._config.llm_provider in ("openai", "anthropic", "local"):
            self._llm_client = object()
            return True

        self._llm_client = None
        return False

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
            if self._llm_client and self._config.llm_api_key:
                print("[FeedbackLayer] Using LLM for feedback generation")
                prompt = self._build_llm_prompt(context, user_state)
                raw = await self._call_llm(prompt)  # make async
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
                ),
            )
        except Exception:
            print("[FeedbackLayer] Error during feedback generation, using fallback")
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

    def _compute_cache_key(self, context: CodeContext) -> str:
        """
        Use a hash of relevant context, not just file name.
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

    def _build_llm_prompt(
        self,
        context: CodeContext,
        user_state: Optional[UserStateEstimate] = None,
    ) -> str:
        """
        Force strict JSON output. This makes parsing reliable.
        """
        cursor = context.cursor_position
        diagnostics = context.diagnostics or []
        diag_text = "\n".join(
            f"- ({getattr(d, 'severity', '')}) {getattr(d, 'message', '')}"
            for d in diagnostics[:8]
        )

        code = (context.file_content or "")[:12000]  # cap prompt size

        return (
            "You are an assistant that generates concise, actionable code feedback for a VS Code user.\n"
            "Return ONLY valid JSON with this schema:\n"
            "{\n"
            '  "items": [\n'
            "    {\n"
            '      "title": string,\n'
            '      "message": string,\n'
            '      "type": "hint"|"suggestion"|"warning"|"explanation"|"simplification",\n'
            '      "priority": "low"|"medium"|"high"\n'
            "    }\n"
            "  ]\n"
            "}\n"
            f"Constraints:\n- max items: {self._config.max_feedback_items}\n"
            f"- max message length per item: {self._config.max_message_length}\n"
            "- Focus on the most likely issue near the cursor.\n\n"
            f"Language: {context.language_id}\n"
            f"File: {context.file_path}\n"
            f"Cursor: line {cursor.line}, col {cursor.character}\n\n"
            f"Diagnostics:\n{diag_text or '(none)'}\n\n"
            "Code:\n"
            "```"
            f"{context.language_id}\n{code}\n```"
        )

    async def _call_llm(self, prompt: str) -> str:
        """
        PoC: keep as async so you can plug in network call later.
        For now, raise if not configured.
        """
        if not (self._llm_client and self._config.llm_api_key):
            raise RuntimeError("LLM not configured")

        # TODO: replace with real provider call (openai/anthropic/local)
        await asyncio.sleep(0.05)
        return json.dumps({
            "items": [
                {
                    "title": "Sample Feedback",
                    "message": "This is a sample feedback item.",
                    "type": "hint",
                    "priority": "medium",
                }
            ]
        })

    def _parse_llm_response(self, response: str, context: CodeContext) -> List[FeedbackItem]:
        """
        Parse strict JSON. If it fails, fallback gracefully.
        """
        try:
            obj = json.loads(response)
            raw_items = obj.get("items", [])[: self._config.max_feedback_items]
            items: List[FeedbackItem] = []

            for it in raw_items:
                title = str(it.get("title", "Feedback"))[:80]
                msg = str(it.get("message", ""))[: self._config.max_message_length]

                # If your FeedbackItem includes type/priority fields, map them here.
                # Otherwise, ignore them for now.
                items.append(FeedbackItem(title=title, message=msg))

            return items or self._generate_fallback_feedback(context)
        except Exception:
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

    def _create_feedback_metadata(self, cached: bool = False, generation_time_ms: float = 0.0) -> FeedbackMetadata:
        """
        Adapt to your actual FeedbackMetadata fields.
        """
        return FeedbackMetadata(
            cached=cached,
            generation_time_ms=generation_time_ms,
        )

    def _filter_and_rank_feedback(self, items: List[FeedbackItem]) -> List[FeedbackItem]:
        """
        For PoC: keep simple.
        - enforce max items
        - enforce priority filter if you add priority to FeedbackItem later
        """
        items = items[: self._config.max_feedback_items]
        # TODO: if FeedbackItem has priority, apply self._min_priority here
        return items

    def _update_rate_limit_window(self) -> None:
        now = time.time()
        self._rate_limit_window.append(now)
        # prune
        cutoff = now - 60.0
        self._rate_limit_window = [t for t in self._rate_limit_window if t >= cutoff]