/**
 * VS Code Webview API wrapper
 */

// Declare the acquireVsCodeApi function that VS Code injects
declare function acquireVsCodeApi(): VSCodeAPI;

interface VSCodeAPI {
  postMessage(message: WebviewMessage): void;
  getState(): unknown;
  setState(state: unknown): void;
}

export interface WebviewMessage {
  type: string;
  payload?: unknown;
}

export interface VSCodeMessage {
  type: string;
  payload: unknown;
}

class VSCodeAPIWrapper {
  private readonly vsCodeApi: VSCodeAPI | undefined;

  constructor() {
    // Check if we're running in a VS Code webview
    if (typeof acquireVsCodeApi === "function") {
      this.vsCodeApi = acquireVsCodeApi();
    }
  }

  /**
   * Post a message to the extension
   */
  public postMessage(message: WebviewMessage): void {
    if (this.vsCodeApi) {
      this.vsCodeApi.postMessage(message);
    } else {
      console.log("[Webview] Message:", message);
    }
  }

  /**
   * Get the persisted state
   */
  public getState(): unknown {
    if (this.vsCodeApi) {
      return this.vsCodeApi.getState();
    }
    return undefined;
  }

  /**
   * Set the persisted state
   */
  public setState<T>(state: T): T {
    if (this.vsCodeApi) {
      this.vsCodeApi.setState(state);
    }
    return state;
  }
}

// Export a singleton instance
export const vscode = new VSCodeAPIWrapper();
