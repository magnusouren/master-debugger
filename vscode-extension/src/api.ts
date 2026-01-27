import * as http from "http";
import * as https from "https";

/**
 * Fetch JSON from the backend /status endpoint.
 */
export async function fetchStatus(host: string, port: number, useHttps = false): Promise<any> {
    const lib = useHttps ? https : http;
    const protocol = useHttps ? "https" : "http";
    const url = `${protocol}://${host}:${port}/status`;

    return new Promise((resolve, reject) => {
        lib.get(url, (res: any) => {
            let raw = "";
            res.setEncoding("utf8");
            res.on("data", (chunk: string) => (raw += chunk));
            res.on("end", () => {
                try {
                    const parsed = JSON.parse(raw);
                    resolve(parsed);
                } catch (err) {
                    reject(err);
                }
            });
        }).on("error", (err: Error) => reject(err));
    });
}
