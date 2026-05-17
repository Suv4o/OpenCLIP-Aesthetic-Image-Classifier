export interface SearchHit {
  filename: string;
  category: string | null;
  confidence: number | null;
  distance: number;
  url: string;
}

export async function searchByText(query: string, limit = 12): Promise<SearchHit[]> {
  const res = await fetch("/api/search/text", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, limit }),
  });
  if (!res.ok) throw new Error(`text search failed: ${res.status} ${await res.text()}`);
  return res.json();
}

export async function searchByImage(file: File, limit = 12): Promise<SearchHit[]> {
  const form = new FormData();
  form.append("file", file);
  form.append("limit", String(limit));
  const res = await fetch("/api/search/image", { method: "POST", body: form });
  if (!res.ok) throw new Error(`image search failed: ${res.status} ${await res.text()}`);
  return res.json();
}
