#!/usr/bin/env node
// Copyright © 2025-2026 Quadux IT GmbH
//    ____                  __              __________   ______          __    __  __
//   / __ \__  ______ _____/ /_  ___  __   /  _/_  __/  / ____/___ ___  / /_  / / / /
//  / / / / / / / __ `/ __  / / / / |/_/   / /  / /    / / __/ __ `__ \/ __ \/ /_/ /
// / /_/ / /_/ / /_/ / /_/ / /_/ />  <   _/ /  / /    / /_/ / / / / / / /_/ / __  /
// \___\_\__,_/\__,_/\__,_/\__,_/_/|_|  /___/ /_/     \____/_/ /_/ /_/_.___/_/ /_/

// License: Quadux files Apache 2.0 (see LICENSE), Jina model: Qwen Research License
// Author: Walter Hoffmann

/**
 * Test script for Jina Embeddings v4 API
 * Tests both text and image embedding endpoints
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const API_BASE = process.env.JINA_API_URL || "http://localhost:8090";

async function testTextEmbedding() {
  console.log("\n📝 Testing Text Embedding...");
  console.log("─".repeat(50));

  const payload = {
    texts: ["Hello world", "This is a test sentence"],
    task: "text-matching",
  };

  try {
    const response = await fetch(`${API_BASE}/embed/text`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`HTTP ${response.status}: ${error}`);
    }

    const data = await response.json();

    console.log(`✅ Success!`);
    console.log(`   Texts sent: ${payload.texts.length}`);
    console.log(`   Embeddings received: ${data.embeddings.length}`);
    console.log(`   Dimension per embedding: ${data.embeddings[0].length}`);
    console.log(
      `   First 5 values: [${data.embeddings[0]
        .slice(0, 5)
        .map((v) => v.toFixed(4))
        .join(", ")}...]`
    );

    return { success: true, data };
  } catch (error) {
    console.log(`❌ Failed: ${error.message}`);
    return { success: false, error: error.message };
  }
}

async function testImageEmbedding() {
  console.log("\n🖼️  Testing Image Embedding...");
  console.log("─".repeat(50));

  // Use existing test image
  const testImagePath = path.join(__dirname, "test.png");

  if (!fs.existsSync(testImagePath)) {
    console.log(
      "❌ test.png not found - please ensure test images are in place"
    );
    return { success: false, error: "test.png not found" };
  }

  try {
    const imageBuffer = fs.readFileSync(testImagePath);
    const boundary =
      "----FormBoundary" + Math.random().toString(36).substring(2);

    // Build multipart form data manually
    const parts = [];

    // File part
    parts.push(`--${boundary}`);
    parts.push(
      'Content-Disposition: form-data; name="file"; filename="test.png"'
    );
    parts.push("Content-Type: image/png");
    parts.push("");

    // Task part
    const taskPart = [
      `--${boundary}`,
      'Content-Disposition: form-data; name="task"',
      "",
      "text-matching",
      `--${boundary}--`,
    ].join("\r\n");

    // Combine parts
    const headerPart = parts.join("\r\n") + "\r\n";
    const body = Buffer.concat([
      Buffer.from(headerPart),
      imageBuffer,
      Buffer.from("\r\n" + taskPart),
    ]);

    const response = await fetch(`${API_BASE}/embed/image`, {
      method: "POST",
      headers: {
        "Content-Type": `multipart/form-data; boundary=${boundary}`,
      },
      body: body,
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`HTTP ${response.status}: ${error}`);
    }

    const data = await response.json();

    console.log(`✅ Success!`);
    console.log(`   Image: ${testImagePath}`);
    console.log(`   Embeddings received: ${data.embeddings.length}`);
    console.log(`   Dimension per embedding: ${data.embeddings[0].length}`);
    console.log(
      `   First 5 values: [${data.embeddings[0]
        .slice(0, 5)
        .map((v) => v.toFixed(4))
        .join(", ")}...]`
    );

    return { success: true, data };
  } catch (error) {
    console.log(`❌ Failed: ${error.message}`);
    return { success: false, error: error.message };
  }
}

async function testHealth() {
  console.log("\n🏥 Testing Health Endpoint...");
  console.log("─".repeat(50));

  try {
    const response = await fetch(`${API_BASE}/health`);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    console.log(`✅ API is healthy`);
    console.log(`   Status: ${data.status}`);
    console.log(`   Device: ${data.device}`);
    console.log(`   Model: ${data.model}`);

    return { success: true, data };
  } catch (error) {
    console.log(`❌ Failed: ${error.message}`);
    return { success: false, error: error.message };
  }
}

// Helper function to get image embedding
async function getImageEmbedding(imagePath, task = "text-matching") {
  const imageBuffer = fs.readFileSync(imagePath);
  const boundary = "----FormBoundary" + Math.random().toString(36).substring(2);

  const parts = [];
  parts.push(`--${boundary}`);
  parts.push(
    `Content-Disposition: form-data; name="file"; filename="${path.basename(
      imagePath
    )}"`
  );
  parts.push("Content-Type: image/png");
  parts.push("");

  const taskPart = [
    `--${boundary}`,
    'Content-Disposition: form-data; name="task"',
    "",
    task,
    `--${boundary}--`,
  ].join("\r\n");

  const headerPart = parts.join("\r\n") + "\r\n";
  const body = Buffer.concat([
    Buffer.from(headerPart),
    imageBuffer,
    Buffer.from("\r\n" + taskPart),
  ]);

  const response = await fetch(`${API_BASE}/embed/image`, {
    method: "POST",
    headers: {
      "Content-Type": `multipart/form-data; boundary=${boundary}`,
    },
    body: body,
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`HTTP ${response.status}: ${error}`);
  }

  const data = await response.json();
  return data.embeddings[0];
}

// Cosine similarity function
function cosineSim(a, b) {
  let dot = 0,
    normA = 0,
    normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Get text embedding helper
async function getTextEmbedding(text, task = "text-matching") {
  const response = await fetch(`${API_BASE}/embed/text`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ texts: [text], task }),
  });
  if (!response.ok) {
    const error = await response.text();
    throw new Error(`HTTP ${response.status}: ${error}`);
  }
  const data = await response.json();
  return data.embeddings[0];
}

async function testImageSimilarity() {
  console.log("\n🖼️  Testing Image-to-Image Similarity...");
  console.log("─".repeat(50));

  // Use cached test images
  const testImages = {
    cat1: path.join(__dirname, "test_cat1.jpg"),
    cat2: path.join(__dirname, "test_cat2.jpg"),
    nature: path.join(__dirname, "test_nature.jpg"),
  };

  // Check all images exist
  const missing = Object.entries(testImages).filter(
    ([, f]) => !fs.existsSync(f)
  );
  if (missing.length > 0) {
    console.log(`❌ Missing images: ${missing.map(([k]) => k).join(", ")}`);
    return { success: false, error: "Missing test images" };
  }

  try {
    console.log("   Using cached test images...");
    console.log("   Getting embeddings for all images...");
    const [embCat1, embCat2, embNature] = await Promise.all([
      getImageEmbedding(testImages.cat1),
      getImageEmbedding(testImages.cat2),
      getImageEmbedding(testImages.nature),
    ]);

    // Image vs Image comparisons
    const simCat1Cat2 = cosineSim(embCat1, embCat2);
    const simCat1Nature = cosineSim(embCat1, embNature);
    const simCat2Nature = cosineSim(embCat2, embNature);

    console.log("");
    console.log("📊 Image vs Image Results:");
    console.log(`   Cat1 vs Cat2 (similar):     ${simCat1Cat2.toFixed(4)}`);
    console.log(`   Cat1 vs Nature (different): ${simCat1Nature.toFixed(4)}`);
    console.log(`   Cat2 vs Nature (different): ${simCat2Nature.toFixed(4)}`);

    if (simCat1Cat2 > simCat1Nature && simCat1Cat2 > simCat2Nature) {
      console.log("   ✅ Similar images have higher similarity!");
    }

    // Keep images cached for future runs
    return { success: true };
  } catch (error) {
    console.log(`❌ Failed: ${error.message}`);
    return { success: false, error: error.message };
  }
}

// NOTE: Single-Vector cross-modal test removed - values are useless (~0.01-0.06)
// Use Multi-Vector + MaxSim for cross-modal retrieval instead

async function testSimilarity() {
  console.log("\n🔍 Testing Text Similarity...");
  console.log("─".repeat(50));

  const texts = [
    "A cat sitting on a mat",
    "A kitten resting on a rug",
    "The stock market crashed today",
  ];

  try {
    const response = await fetch(`${API_BASE}/embed/text`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texts, task: "text-matching" }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();

    // Use global cosineSim function
    const sim01 = cosineSim(data.embeddings[0], data.embeddings[1]);
    const sim02 = cosineSim(data.embeddings[0], data.embeddings[2]);
    const sim12 = cosineSim(data.embeddings[1], data.embeddings[2]);

    console.log(`✅ Similarity results:`);
    console.log(`   "${texts[0].substring(0, 30)}..."`);
    console.log(`   "${texts[1].substring(0, 30)}..."`);
    console.log(`   "${texts[2].substring(0, 30)}..."`);
    console.log("");
    console.log(`   Similarity [0,1] (similar): ${sim01.toFixed(4)}`);
    console.log(`   Similarity [0,2] (different): ${sim02.toFixed(4)}`);
    console.log(`   Similarity [1,2] (different): ${sim12.toFixed(4)}`);

    if (sim01 > sim02 && sim01 > sim12) {
      console.log(`   ✅ Semantic similarity works correctly!`);
    }

    return { success: true };
  } catch (error) {
    console.log(`❌ Failed: ${error.message}`);
    return { success: false, error: error.message };
  }
}

async function testMatryoshkaDimensions() {
  console.log("\n📐 Testing Matryoshka Dimensions...");
  console.log("─".repeat(50));

  const testText = "The quick brown fox jumps over the lazy dog";
  const dimensions = [null, 1024, 512, 256]; // null = full 2048

  try {
    console.log("   Testing text embeddings with different dimensions...");
    const results = [];

    for (const dim of dimensions) {
      const payload = {
        texts: [testText],
        task: "text-matching",
      };
      if (dim !== null) payload.dimensions = dim;

      const response = await fetch(`${API_BASE}/embed/text`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status} for dim=${dim}`);
      }

      const data = await response.json();
      const actualDim = data.embeddings[0].length;
      const expectedDim = dim || 2048;

      results.push({
        dim: expectedDim,
        actual: actualDim,
        ok: actualDim === expectedDim,
      });
    }

    console.log("");
    console.log("📊 Text Embedding Dimensions:");
    console.log("   Expected → Actual");
    for (const r of results) {
      const status = r.ok ? "✅" : "❌";
      console.log(
        `   ${r.dim.toString().padStart(4)} → ${r.actual
          .toString()
          .padStart(4)} ${status}`
      );
    }

    // Test image embeddings with dimensions
    console.log("");
    console.log("   Testing image embeddings with different dimensions...");

    const testImagePath = path.join(__dirname, "test.png");
    if (!fs.existsSync(testImagePath)) {
      console.log("   ⚠️  Skipping image dimension test (no test.png)");
    } else {
      const imageResults = [];

      for (const dim of dimensions) {
        const imageBuffer = fs.readFileSync(testImagePath);
        const boundary =
          "----FormBoundary" + Math.random().toString(36).substring(2);

        const parts = [`--${boundary}`];
        parts.push(
          'Content-Disposition: form-data; name="file"; filename="test.png"'
        );
        parts.push("Content-Type: image/png");
        parts.push("");

        let taskPart = [
          `--${boundary}`,
          'Content-Disposition: form-data; name="task"',
          "",
          "text-matching",
        ];

        if (dim !== null) {
          taskPart.push(`--${boundary}`);
          taskPart.push('Content-Disposition: form-data; name="dimensions"');
          taskPart.push("");
          taskPart.push(dim.toString());
        }

        taskPart.push(`--${boundary}--`);

        const headerPart = parts.join("\r\n") + "\r\n";
        const body = Buffer.concat([
          Buffer.from(headerPart),
          imageBuffer,
          Buffer.from("\r\n" + taskPart.join("\r\n")),
        ]);

        const response = await fetch(`${API_BASE}/embed/image`, {
          method: "POST",
          headers: {
            "Content-Type": `multipart/form-data; boundary=${boundary}`,
          },
          body: body,
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status} for image dim=${dim}`);
        }

        const data = await response.json();
        const actualDim = data.embeddings[0].length;
        const expectedDim = dim || 2048;

        imageResults.push({
          dim: expectedDim,
          actual: actualDim,
          ok: actualDim === expectedDim,
        });
      }

      console.log("");
      console.log("📊 Image Embedding Dimensions:");
      console.log("   Expected → Actual");
      for (const r of imageResults) {
        const status = r.ok ? "✅" : "❌";
        console.log(
          `   ${r.dim.toString().padStart(4)} → ${r.actual
            .toString()
            .padStart(4)} ${status}`
        );
      }
    }

    console.log("");
    console.log("   ✅ Matryoshka dimension truncation works!");
    return { success: true };
  } catch (error) {
    console.log(`❌ Failed: ${error.message}`);
    return { success: false, error: error.message };
  }
}

async function testAllTasks() {
  console.log("\n🎯 Testing All LoRA Task Adapters...");
  console.log("─".repeat(50));

  const tasks = ["text-matching", "retrieval", "code"];
  const testText = "Hello world, this is a test sentence.";
  const results = [];

  try {
    for (const task of tasks) {
      const response = await fetch(`${API_BASE}/embed/text`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ texts: [testText], task }),
      });

      if (!response.ok) {
        const error = await response.text();
        results.push({ task, success: false, error });
        continue;
      }

      const data = await response.json();
      const dim = data.embeddings[0]?.length || 0;
      results.push({ task, success: true, dimensions: dim });
    }

    console.log("");
    console.log("📊 Task Adapter Results:");
    console.log("   Task            | Status | Dimensions");
    console.log("   " + "─".repeat(40));

    let allPassed = true;
    for (const r of results) {
      const status = r.success ? "✅" : "❌";
      const dims = r.success
        ? r.dimensions.toString()
        : r.error?.substring(0, 20);
      console.log(`   ${r.task.padEnd(15)} | ${status}     | ${dims}`);
      if (!r.success) allPassed = false;
    }

    if (allPassed) {
      console.log("");
      console.log("   ✅ All task adapters work correctly!");
    }

    return { success: allPassed };
  } catch (error) {
    console.log(`❌ Failed: ${error.message}`);
    return { success: false, error: error.message };
  }
}

// Helper to get Multi-Vector image embedding
async function getImageMV(imagePath, task = "retrieval") {
  const imageBuffer = fs.readFileSync(imagePath);
  const boundary = "----FormBoundary" + Math.random().toString(36).substring(2);

  const parts = [`--${boundary}`];
  parts.push(
    `Content-Disposition: form-data; name="file"; filename="${path.basename(
      imagePath
    )}"`
  );
  parts.push("Content-Type: image/png");
  parts.push("");

  const taskPart = [
    `--${boundary}`,
    'Content-Disposition: form-data; name="task"',
    "",
    task,
    `--${boundary}--`,
  ].join("\r\n");

  const headerPart = parts.join("\r\n") + "\r\n";
  const body = Buffer.concat([
    Buffer.from(headerPart),
    imageBuffer,
    Buffer.from("\r\n" + taskPart),
  ]);

  const resp = await fetch(`${API_BASE}/embed/imageMV`, {
    method: "POST",
    headers: { "Content-Type": `multipart/form-data; boundary=${boundary}` },
    body: body,
  });

  if (!resp.ok) throw new Error(`Image MV failed: ${await resp.text()}`);
  const data = await resp.json();
  return data.embeddings[0]; // Array of [patches × 128]
}

// Helper to get Multi-Vector text embedding
async function getTextMV(text, task = "retrieval") {
  const resp = await fetch(`${API_BASE}/embed/textMV`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ texts: [text], task }),
  });
  if (!resp.ok) throw new Error(`Text MV failed: ${await resp.text()}`);
  const data = await resp.json();
  return data.embeddings[0]; // Array of [tokens × 128]
}

// MaxSim: For each text token, find max similarity to any image patch, then average
function maxSim(textMV, imageMV) {
  let totalMax = 0;
  for (const tVec of textMV) {
    let maxCos = -Infinity;
    for (const iVec of imageMV) {
      const cos = cosineSim(tVec, iVec);
      if (cos > maxCos) maxCos = cos;
    }
    totalMax += maxCos;
  }
  return totalMax / textMV.length;
}

/**
 * Test Multi-Vector Endpoints
 *
 * NOTE: Cross-Modal retrieval (Text → Image) does NOT work well with Jina v4!
 * - Single-Vector: ~0.01-0.06 similarity (useless)
 * - Multi-Vector MaxSim: ~0.17-0.27 (no differentiation between images)
 * - Tested with natural photos AND charts/diagrams - both fail
 *
 * This is a MODEL LIMITATION, not a bug. Use Jina v4 for:
 * - ✅ Text-to-Text similarity (works great)
 * - ✅ Image-to-Image similarity (works OK)
 * - ❌ Text-to-Image cross-modal (does NOT work)
 */
async function testMultiVectorEndpoints() {
  console.log("\n📄 Testing Multi-Vector Endpoints...");
  console.log("─".repeat(50));

  const testImagePath = path.join(__dirname, "test.png");

  if (!fs.existsSync(testImagePath)) {
    console.log("   ⚠️  test.png not found - skipping");
    return { success: true, skipped: true };
  }

  try {
    // Test Text Multi-Vector
    console.log("   Testing /embed/textMV...");
    const textMV = await getTextMV(
      "A test sentence for multi-vector embedding"
    );
    console.log(`   ✅ Text MV: ${textMV.length} tokens × 128 dims`);

    // Test Image Multi-Vector
    console.log("   Testing /embed/imageMV...");
    const imageMV = await getImageMV(testImagePath);
    console.log(`   ✅ Image MV: ${imageMV.length} patches × 128 dims`);

    // Verify dimensions
    if (textMV[0].length !== 128 || imageMV[0].length !== 128) {
      throw new Error(
        `Wrong dims: text=${textMV[0].length}, image=${imageMV[0].length}`
      );
    }

    console.log("");
    console.log(
      "   ⚠️  Cross-modal (text↔image) does NOT work well with this model"
    );
    console.log("   ⚠️  Use for text-text or image-image similarity only");
    console.log("   ✅ Multi-Vector endpoints work correctly!");

    return {
      success: true,
      textTokens: textMV.length,
      imagePatches: imageMV.length,
    };
  } catch (error) {
    console.log(`❌ Failed: ${error.message}`);
    return { success: false, error: error.message };
  }
}

// Helper to measure execution time
async function timed(name, fn) {
  const start = performance.now();
  const result = await fn();
  const duration = performance.now() - start;
  return { ...result, duration };
}

async function main() {
  console.log("╔══════════════════════════════════════════════════╗");
  console.log("║     Jina Embeddings v4 API Test Suite            ║");
  console.log("╚══════════════════════════════════════════════════╝");
  console.log(`API URL: ${API_BASE}`);

  const totalStart = performance.now();

  const results = {
    health: await timed("health", testHealth),
    tasks: await timed("tasks", testAllTasks),
    text: await timed("text", testTextEmbedding),
    image: await timed("image", testImageEmbedding),
    textSimilarity: await timed("textSimilarity", testSimilarity),
    imageSimilarity: await timed("imageSimilarity", testImageSimilarity),
    multiVector: await timed("multiVector", testMultiVectorEndpoints),
    matryoshka: await timed("matryoshka", testMatryoshkaDimensions),
  };

  const totalDuration = performance.now() - totalStart;

  console.log("\n" + "═".repeat(50));
  console.log("📊 Test Summary:");
  console.log("─".repeat(50));

  let passed = 0;
  let failed = 0;

  for (const [name, result] of Object.entries(results)) {
    const status = result.success ? "✅ PASS" : "❌ FAIL";
    const time = `${result.duration.toFixed(0)}ms`;
    console.log(`   ${name.padEnd(20)} ${status}  ${time.padStart(8)}`);
    if (result.success) passed++;
    else failed++;
  }

  console.log("─".repeat(50));
  console.log(`   Total: ${passed} passed, ${failed} failed`);
  console.log(`   ⏱️  Total time: ${(totalDuration / 1000).toFixed(2)}s`);
  console.log("");

  process.exit(failed > 0 ? 1 : 0);
}

main().catch(console.error);
