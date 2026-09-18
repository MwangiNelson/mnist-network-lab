import { PointerEvent, useEffect, useRef, useState } from "react";

type Probability = { digit: number; probability: number };
type Prediction = {
  digit: number;
  confidence: number;
  probabilities: Probability[];
  model: string;
};

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8010";
const CANVAS_SIZE = 280;

function preparePixels(source: HTMLCanvasElement): number[] | null {
  const context = source.getContext("2d", { willReadFrequently: true });
  if (!context) return null;
  const data = context.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE).data;
  let minX = CANVAS_SIZE;
  let minY = CANVAS_SIZE;
  let maxX = -1;
  let maxY = -1;

  for (let y = 0; y < CANVAS_SIZE; y += 1) {
    for (let x = 0; x < CANVAS_SIZE; x += 1) {
      const value = data[(y * CANVAS_SIZE + x) * 4];
      if (value > 20) {
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
      }
    }
  }
  if (maxX < minX || maxY < minY) return null;

  const width = maxX - minX + 1;
  const height = maxY - minY + 1;
  const scale = 20 / Math.max(width, height);
  const drawWidth = Math.max(1, Math.round(width * scale));
  const drawHeight = Math.max(1, Math.round(height * scale));
  const target = document.createElement("canvas");
  target.width = 28;
  target.height = 28;
  const targetContext = target.getContext("2d", { willReadFrequently: true });
  if (!targetContext) return null;
  targetContext.imageSmoothingEnabled = true;

  const render = (offsetX: number, offsetY: number) => {
    targetContext.fillStyle = "black";
    targetContext.fillRect(0, 0, 28, 28);
    targetContext.drawImage(
      source,
      minX,
      minY,
      width,
      height,
      offsetX,
      offsetY,
      drawWidth,
      drawHeight,
    );
    return targetContext.getImageData(0, 0, 28, 28).data;
  };

  // MNIST fits each digit into a 20-pixel box and then positions it so its
  // center of mass lands in the middle of the 28 by 28 field, not its bounding
  // box. The two differ by a few pixels on any digit whose ink is unevenly
  // distributed, a 7 or a 1 most of all. A fully connected model has no
  // translation invariance, so those few pixels move every one of the 784
  // inputs. On 1,000 MNIST test digits pushed through this pipeline, bounding
  // box centring scored 87.4% and read only 61.6% of the 7s correctly; centre
  // of mass scored 96.9%, against 98.1% for the raw digits. See
  // tools/check_canvas_centring.py.
  const centeredX = (28 - drawWidth) / 2;
  const centeredY = (28 - drawHeight) / 2;
  const firstPass = render(centeredX, centeredY);

  let mass = 0;
  let momentX = 0;
  let momentY = 0;
  for (let y = 0; y < 28; y += 1) {
    for (let x = 0; x < 28; x += 1) {
      const value = firstPass[(y * 28 + x) * 4];
      mass += value;
      momentX += value * x;
      momentY += value * y;
    }
  }
  if (mass === 0) return null;

  // 13.5 is the middle of a 0..27 grid.
  const reduced = render(
    centeredX + (13.5 - momentX / mass),
    centeredY + (13.5 - momentY / mass),
  );
  return Array.from({ length: 784 }, (_, index) => reduced[index * 4] / 255);
}

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drawing = useRef(false);
  const previous = useRef({ x: 0, y: 0 });
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "error">("idle");
  const [message, setMessage] = useState("Draw one digit, then run the model.");

  useEffect(() => {
    clearCanvas();
  }, []);

  function clearCanvas() {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;
    context.fillStyle = "#000";
    context.fillRect(0, 0, canvas.width, canvas.height);
    setPrediction(null);
    setStatus("idle");
    setMessage("Draw one digit, then run the model.");
  }

  function point(event: PointerEvent<HTMLCanvasElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    return {
      x: ((event.clientX - rect.left) / rect.width) * CANVAS_SIZE,
      y: ((event.clientY - rect.top) / rect.height) * CANVAS_SIZE,
    };
  }

  function startDrawing(event: PointerEvent<HTMLCanvasElement>) {
    event.currentTarget.setPointerCapture(event.pointerId);
    drawing.current = true;
    previous.current = point(event);
  }

  function draw(event: PointerEvent<HTMLCanvasElement>) {
    if (!drawing.current) return;
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!context) return;
    const current = point(event);
    context.strokeStyle = "#fff";
    context.lineWidth = 24;
    context.lineCap = "round";
    context.lineJoin = "round";
    context.beginPath();
    context.moveTo(previous.current.x, previous.current.y);
    context.lineTo(current.x, current.y);
    context.stroke();
    previous.current = current;
  }

  function stopDrawing() {
    drawing.current = false;
  }

  async function classify() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const pixels = preparePixels(canvas);
    if (!pixels) {
      setStatus("error");
      setMessage("The canvas is empty. Draw a digit first.");
      return;
    }
    setStatus("loading");
    setMessage("Running the fully connected model...");
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pixels }),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail ?? `API returned ${response.status}`);
      }
      const result: Prediction = await response.json();
      setPrediction(result);
      setStatus("idle");
      setMessage(`The model reads this as ${result.digit}.`);
    } catch (error) {
      setStatus("error");
      setMessage(error instanceof Error ? error.message : "Prediction failed.");
    }
  }

  const probabilities = prediction?.probabilities ??
    Array.from({ length: 10 }, (_, digit) => ({ digit, probability: 0 }));

  return (
    <main>
      <header className="masthead">
        <a className="brand" href="/" aria-label="MNIST network lab home">
          MNIST / network lab
        </a>
        <a href="https://github.com/MwangiNelson/mnist-network-lab">View the notebook</a>
      </header>

      <section className="intro">
        <h1>Write a digit.<br />See what the network learned.</h1>
        <p>
          This is the student-designed fully connected model from the DSA 8401
          assignment. It reads 784 normalized pixels and returns ten probabilities.
        </p>
      </section>

      <section className="workbench" aria-label="Digit classifier">
        <div className="drawing-panel">
          <div className="panel-heading">
            <h2>Drawing pad</h2>
            <button className="text-button" type="button" onClick={clearCanvas}>Clear</button>
          </div>
          <canvas
            ref={canvasRef}
            width={CANVAS_SIZE}
            height={CANVAS_SIZE}
            onPointerDown={startDrawing}
            onPointerMove={draw}
            onPointerUp={stopDrawing}
            onPointerCancel={stopDrawing}
            aria-label="Draw a handwritten digit from zero to nine"
          />
          <button className="run-button" type="button" onClick={classify} disabled={status === "loading"}>
            {status === "loading" ? "Running model" : "Classify drawing"}
          </button>
          <p className={status === "error" ? "status error" : "status"} role="status">{message}</p>
        </div>

        <div className="result-panel" aria-live="polite">
          <div className="result-number">{prediction?.digit ?? "?"}</div>
          <p className="confidence">
            {prediction ? `${(prediction.confidence * 100).toFixed(1)}% confidence` : "Awaiting a drawing"}
          </p>
          <div className="probabilities">
            {probabilities.sort((a, b) => a.digit - b.digit).map((item) => (
              <div className="probability" key={item.digit}>
                <span>{item.digit}</span>
                <div className="track"><div style={{ width: `${item.probability * 100}%` }} /></div>
                <output>{(item.probability * 100).toFixed(1)}%</output>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="method">
        <h2>What happens to the drawing</h2>
        <ol>
          <li>The browser finds the stroke bounds and fits them into a 20-pixel box.</li>
          <li>It centers the digit on a 28 by 28 black field and scales pixels to 0 through 1.</li>
          <li>The saved Keras model returns a probability for each digit.</li>
        </ol>
        <p className="privacy">No account. No stored drawings. This public demo is for coursework, not handwriting verification.</p>
      </section>
    </main>
  );
}
