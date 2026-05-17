export interface PointData {
  filename: string;
  category: string | null;
  confidence: number | null;
  x: number;
  y: number;
  z: number;
}

export async function loadCoords(): Promise<PointData[]> {
  const res = await fetch("/coords.json");
  if (!res.ok) {
    throw new Error(
      `Could not load coords.json (${res.status}). Did you run \`uv run generate_coords.py\` yet?`
    );
  }
  return res.json();
}
