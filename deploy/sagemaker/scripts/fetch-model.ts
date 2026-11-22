/**
 * Download an HF Hub model snapshot for SageMaker ModelDataSource.
 *
 *     npx tsx scripts/fetch-model.ts MODEL REVISION OUT_DIR
 *
 * Writes a flat Hugging Face layout under OUT_DIR. SageMaker mounts that
 * prefix at /opt/ml/model, so the container can load offline.
 */
import { cp } from "node:fs/promises";
import * as path from "node:path";
import { snapshotDownload } from "@huggingface/hub";

async function main(model: string, revision: string, outDir: string): Promise<void> {
  const snapshot = await snapshotDownload({ repo: model, revision });
  console.error(`model ${model}@${path.basename(snapshot)} -> ${outDir}`);
  await cp(snapshot, outDir, { recursive: true, dereference: true });
}

const [model, revision, outDir] = process.argv.slice(2);
if (!model || !revision || !outDir) {
  console.error("usage: npx tsx scripts/fetch-model.ts MODEL REVISION OUT_DIR");
  process.exit(1);
}
main(model, revision, outDir).catch((err) => {
  console.error(err);
  process.exit(1);
});
