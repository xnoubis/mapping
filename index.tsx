
import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { createRoot } from 'react-dom/client';

// ============================================================================
// MATH & UTILS
// ============================================================================

class SeededRNG {
  private seed: number;
  constructor(seed: number) { this.seed = seed; }
  
  next(): number {
    this.seed = (this.seed * 9301 + 49297) % 233280;
    return this.seed / 233280;
  }
  
  gauss(mean: number, std: number): number {
    let u = 0, v = 0;
    while(u === 0) u = this.next();
    while(v === 0) v = this.next();
    const standard = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    return mean + standard * std;
  }
}

function simpleHash(str: string): number {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash |= 0;
  }
  return hash;
}

function computeEmbedding(text: string, dim: number = 768): number[] {
  const seed = Math.abs(simpleHash(text));
  const rng = new SeededRNG(seed);
  const embedding: number[] = [];
  
  const charSum = text.slice(0, 100).split('').reduce((acc, c) => acc + c.charCodeAt(0), 0);

  for (let i = 0; i < dim; i++) {
    const base = rng.gauss(0, 1);
    const textBias = Math.sin(charSum * 0.01 + i * 0.1) * 0.3;
    embedding.push(base + textBias);
  }
  
  const mag = Math.sqrt(embedding.reduce((acc, x) => acc + x*x, 0));
  return embedding.map(x => x / (mag + 1e-10));
}

function cosineDistance(a: number[], b: number[]): number {
  let dot = 0, magA = 0, magB = 0;
  for(let i=0; i<a.length; i++) {
    dot += a[i]*b[i];
    magA += a[i]*a[i];
    magB += b[i]*b[i];
  }
  magA = Math.sqrt(magA);
  magB = Math.sqrt(magB);
  if (magA < 1e-10 || magB < 1e-10) return 1.0;
  return 1.0 - (dot / (magA * magB));
}

function getDragonTurn(step: number): number {
  if (step <= 0) return 1;
  return (((step & -step) << 1) & step) === 0 ? 1 : -1;
}

function computeFractalDepth(step: number): number {
  if (step <= 0) return 0;
  let depth = 0;
  let s = step;
  while (s % 2 === 0) {
    s = Math.floor(s / 2);
    depth++;
  }
  return depth;
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================

class HueVector {
  constructor(
    public boundaryPressure: number = 0,
    public loopiness: number = 0,
    public novelty: number = 0,
    public coherence: number = 0,
    public risk: number = 0
  ) {}

  magnitude(): number {
    return Math.sqrt(
      this.boundaryPressure**2 + this.loopiness**2 + this.novelty**2 +
      this.coherence**2 + this.risk**2
    );
  }

  decay(delta: number = 0.05): HueVector {
    return new HueVector(
      this.boundaryPressure * (1 - delta),
      this.loopiness * (1 - delta),
      this.novelty * (1 - delta),
      this.coherence * (1 - delta),
      this.risk * (1 - delta)
    );
  }

  update(quality: HueVector, lambda: number = 0.3): HueVector {
    return new HueVector(
      (1-lambda)*this.boundaryPressure + lambda*quality.boundaryPressure,
      (1-lambda)*this.loopiness + lambda*quality.loopiness,
      (1-lambda)*this.novelty + lambda*quality.novelty,
      (1-lambda)*this.coherence + lambda*quality.coherence,
      (1-lambda)*this.risk + lambda*quality.risk
    );
  }
}

interface TerrainNode {
  id: string;
  content: string;
  embedding: number[];
  hue: HueVector;
  visitCount: number;
  // Visual state
  x: number;
  y: number;
  vx: number;
  vy: number;
}

interface SnapEvent {
  deltaEntropy: number;
  stabilitySteps: number;
  evalGain: number;
  structuralDelta: string;
  timestamp: string;
  fractalIteration: number;
  eigenResonance: number;
  dragonSequence: number[];
}

interface StereogramToken {
  id: string; // generated for React key
  operatorHash: string;
  terrainHash: string;
  snapEvent: SnapEvent;
  dragonAttestation: any;
  scores: number[];
  timestamp: string;
}

// ============================================================================
// ROSETTA ENGINE
// ============================================================================

class RosettaEngine {
  dim: number = 768;
  kNeighbors: number = 5; // Low K for sparse terrain in visualization
  
  nodes: Map<string, TerrainNode> = new Map();
  adjacency: Map<string, string[]> = new Map();
  
  currentNodeId: string | null = null;
  visitedPath: string[] = [];
  dragonTurns: number[] = [];
  entropyHistory: number[] = [];
  
  snaps: SnapEvent[] = [];
  tokens: StereogramToken[] = [];
  
  // Params
  alpha = 0.25; // hue
  beta = 0.25;  // similarity
  gamma = 0.25; // novelty
  zeta = 0.25;  // fractal
  
  snapEpsilon = 0.003;
  snapStability = 2;
  snapGainThreshold = 0.01;
  baselineEval = 0.0;
  
  // For visualization: last fractal target position
  lastFractalTarget: number[] | null = null; 

  ingest(text: string) {
    // Reset
    this.nodes.clear();
    this.adjacency.clear();
    this.visitedPath = [];
    this.dragonTurns = [];
    this.entropyHistory = [];
    this.snaps = [];
    this.tokens = [];
    this.baselineEval = 0.0;
    
    // Chunking (simple sentence split for demo)
    const chunks = text.split(/(?<=[.?!])\s+/).filter(c => c.length > 10).slice(0, 50); // Limit to 50 nodes for perf
    
    chunks.forEach((chunk, i) => {
      const id = `node_${i}`;
      const emb = computeEmbedding(chunk, this.dim);
      this.nodes.set(id, {
        id,
        content: chunk.slice(0, 150),
        embedding: emb,
        hue: new HueVector(),
        visitCount: 0,
        // Init visual position randomly
        x: Math.random() * 800,
        y: Math.random() * 600,
        vx: 0,
        vy: 0
      });
    });
    
    this._buildAdjacency();
    
    if (this.nodes.size > 0) {
      this.currentNodeId = this.nodes.keys().next().value;
      this.visitedPath = [this.currentNodeId!];
    }
  }
  
  _buildAdjacency() {
    const nodeIds = Array.from(this.nodes.keys());
    nodeIds.forEach(id => {
      const node = this.nodes.get(id)!;
      const dists: {id: string, d: number}[] = [];
      nodeIds.forEach(otherId => {
        if (id === otherId) return;
        const other = this.nodes.get(otherId)!;
        dists.push({ id: otherId, d: cosineDistance(node.embedding, other.embedding) });
      });
      dists.sort((a, b) => a.d - b.d);
      this.adjacency.set(id, dists.slice(0, this.kNeighbors).map(x => x.id));
    });
  }
  
  step(): { snapped: boolean, token?: StereogramToken } {
    if (!this.currentNodeId) return { snapped: false };
    
    const stepIdx = this.visitedPath.length;
    const turn = getDragonTurn(stepIdx);
    this.dragonTurns.push(turn);
    
    // Navigation
    const nextNodeId = this._selectNextNode(0.8, stepIdx);
    const entropy = this._computeEntropy(stepIdx);
    
    this.visitedPath.push(nextNodeId);
    this.currentNodeId = nextNodeId;
    this.entropyHistory.push(entropy);
    
    const node = this.nodes.get(nextNodeId)!;
    node.visitCount++;
    
    this._updateHue(node);
    const evalScore = this._computeStepValue(node);
    
    // Snap Detection
    const snap = this._checkSnap(evalScore, stepIdx);
    let token: StereogramToken | undefined;
    
    if (snap) {
      this.snaps.push(snap);
      token = this._mintToken(snap, evalScore);
      this.tokens.push(token);
    }
    
    this._applyDecay();
    
    return { snapped: !!snap, token };
  }
  
  _selectNextNode(temperature: number, step: number): string {
    const currentNode = this.nodes.get(this.currentNodeId!)!;
    const neighbors = this.adjacency.get(this.currentNodeId!) || [];
    
    if (neighbors.length === 0) return this.currentNodeId!;
    
    // Fractal Target
    const fractalTarget = this._computeFractalTarget(currentNode.embedding, step);
    this.lastFractalTarget = fractalTarget; // Store for viz
    
    const energies: {id: string, e: number}[] = [];
    
    neighbors.forEach(nid => {
      const node = this.nodes.get(nid)!;
      const hueE = node.hue.magnitude();
      const simE = cosineDistance(currentNode.embedding, node.embedding); // 1 - similarity
      const noveltyE = -1.0 / (1 + node.visitCount);
      const fractalE = cosineDistance(node.embedding, fractalTarget);
      
      const E = (this.alpha * hueE) + (this.beta * simE) + (this.gamma * noveltyE) + (this.zeta * fractalE);
      energies.push({ id: nid, e: E });
    });
    
    // Softmax
    const minE = Math.min(...energies.map(x => x.e));
    const expVals = energies.map(x => ({ id: x.id, v: Math.exp(-(x.e - minE) / temperature) }));
    const total = expVals.reduce((sum, x) => sum + x.v, 0);
    
    const r = Math.random();
    let cumulative = 0;
    for (const item of expVals) {
      cumulative += item.v / total;
      if (r <= cumulative) return item.id;
    }
    return neighbors[neighbors.length - 1];
  }
  
  _computeFractalTarget(currentEmb: number[], step: number): number[] {
    const turn = getDragonTurn(step);
    const depth = computeFractalDepth(step);
    const mag = 0.3 / (1 + depth * 0.2);
    
    const target = [...currentEmb];
    const axis1 = step % this.dim;
    const axis2 = (step * 7) % this.dim;
    
    target[axis1] += turn * mag;
    target[axis2] += turn * mag * 0.5;
    return target;
  }
  
  _computeEntropy(step: number): number {
    const neighbors = this.adjacency.get(this.currentNodeId!) || [];
    if (neighbors.length <= 1) return 0;
    
    const currentNode = this.nodes.get(this.currentNodeId!)!;
    const fractalTarget = this._computeFractalTarget(currentNode.embedding, step);
    
    const energies = neighbors.map(nid => {
      const node = this.nodes.get(nid)!;
      const hueE = node.hue.magnitude();
      const simE = cosineDistance(currentNode.embedding, node.embedding);
      const noveltyE = -1.0 / (1 + node.visitCount);
      const fractalE = cosineDistance(node.embedding, fractalTarget);
      return (this.alpha * hueE) + (this.beta * simE) + (this.gamma * noveltyE) + (this.zeta * fractalE);
    });
    
    const minE = Math.min(...energies);
    const expVals = energies.map(e => Math.exp(-(e - minE) / 0.8));
    const total = expVals.reduce((a, b) => a + b, 0);
    const probs = expVals.map(v => v / total);
    
    return -probs.reduce((sum, p) => p > 0 ? sum + p * Math.log(p) : sum, 0);
  }
  
  _updateHue(node: TerrainNode) {
    const quality = new HueVector(
      0.1,
      Math.min(node.visitCount * 0.2, 1.0),
      1.0 / (1 + node.visitCount),
      0.5,
      0.0
    );
    node.hue = node.hue.update(quality);
  }
  
  _computeStepValue(node: TerrainNode): number {
    const novelty = 0.1 / (1 + node.visitCount * 0.5);
    let coherence = 0.05;
    
    if (this.visitedPath.length > 2) {
      const prevId = this.visitedPath[this.visitedPath.length - 2];
      const prev = this.nodes.get(prevId)!;
      const sim = 1.0 - cosineDistance(node.embedding, prev.embedding);
      coherence = sim * 0.05;
    }
    return novelty + coherence;
  }
  
  _checkSnap(evalScore: number, step: number): SnapEvent | null {
    if (this.entropyHistory.length < this.snapStability + 1) return null;
    
    const recent = this.entropyHistory[this.entropyHistory.length - 1];
    const prior = this.entropyHistory[this.entropyHistory.length - (this.snapStability + 1)];
    const deltaH = prior - recent;
    
    if (deltaH < this.snapEpsilon) return null;
    
    const stableRegion = this.entropyHistory.slice(-this.snapStability);
    const variance = stableRegion.reduce((acc, e) => acc + (e-recent)**2, 0) / stableRegion.length;
    
    if (variance > 0.5) return null;
    
    const gain = evalScore - this.baselineEval;
    if (gain < this.snapGainThreshold) return null;
    
    this.baselineEval = evalScore;
    
    return {
      deltaEntropy: deltaH,
      stabilitySteps: this.snapStability,
      evalGain: gain,
      structuralDelta: `step_${step}`,
      timestamp: new Date().toISOString(),
      fractalIteration: step,
      foldingDepth: computeFractalDepth(step),
      eigenResonance: 0.5, // Placeholder
      dragonSequence: [...this.dragonTurns]
    } as SnapEvent;
  }
  
  _mintToken(snap: SnapEvent, evalScore: number): StereogramToken {
    // Hashes simplified for JS demo
    const hash = simpleHash(snap.timestamp + snap.structuralDelta).toString(16);
    return {
      id: `token_${hash}_${Math.random()}`,
      operatorHash: "rosetta_v2",
      terrainHash: "corpus_terrain_hash",
      snapEvent: snap,
      dragonAttestation: {
        seqLen: snap.dragonSequence.length,
        proof: "zk_mock_proof"
      },
      scores: [evalScore, snap.eigenResonance],
      timestamp: snap.timestamp
    };
  }
  
  _applyDecay(delta: number = 0.03) {
    this.nodes.forEach(node => {
      node.hue = node.hue.decay(delta);
    });
  }

  // Force Directed Simulation Step
  simulatePhysics(width: number, height: number) {
    const k = 50; // Ideal length
    const nodesArr = Array.from(this.nodes.values());
    
    // Repulsion
    for (let i = 0; i < nodesArr.length; i++) {
      for (let j = i + 1; j < nodesArr.length; j++) {
        const a = nodesArr[i];
        const b = nodesArr[j];
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const d2 = dx*dx + dy*dy + 0.1;
        const d = Math.sqrt(d2);
        const f = 5000 / d2;
        const fx = (dx / d) * f;
        const fy = (dy / d) * f;
        
        a.vx -= fx; a.vy -= fy;
        b.vx += fx; b.vy += fy;
      }
    }
    
    // Springs (Adjacency)
    this.adjacency.forEach((neighbors, id) => {
      const a = this.nodes.get(id)!;
      neighbors.forEach(nid => {
        const b = this.nodes.get(nid)!;
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const d = Math.sqrt(dx*dx + dy*dy) || 0.1;
        // The spring strength could be inverse of embedding distance (similarity)
        // For now constant
        const f = (d - k) * 0.05;
        const fx = (dx/d) * f;
        const fy = (dy/d) * f;
        
        a.vx += fx; a.vy += fy;
        b.vx -= fx; b.vy -= fy;
      });
    });
    
    // Center Gravity
    nodesArr.forEach(n => {
      n.vx += (width/2 - n.x) * 0.01;
      n.vy += (height/2 - n.y) * 0.01;
      
      // Update
      n.vx *= 0.9; // damping
      n.vy *= 0.9;
      n.x += n.vx;
      n.y += n.vy;
    });
  }
}

// ============================================================================
// REACT COMPONENTS
// ============================================================================

const DEFAULT_CORPUS = `
The internet's topology becomes navigable terrain - not as content but as constraint-structure. Links are paths, latency is elevation, semantic clusters are geological features.
An agent traverses without naming - it encounters boundaries as mirrors (the constraint tells you what you were looking for by being the place something stops). This is the "attention triangle" applied: partial knowledge, asymmetric visibility, meaning from constraint.
Boids + slime mold memory - the traversal isn't logged, it's hued. Each interaction leaves chromatic residue (like pheromone trails) that encodes the quality of attention, not the content. The internet gains a "contextual fur" - each strand a history of how it was encountered.
The Rosetta Wheel - multiple specialized LLMs as dormant spines. Terrain fold triggers rotation, one wakes, maps, snaps, witnesses, returns to stasis. This prevents monoculture and model collapse.
Stereogram tokens - IP isn't the artifact, it's the constraint configuration that induces the snap. Neither user nor model alone can produce it. The token witnesses co-instantiation conditions, not content.
Share-to-Platform as the attestation bridge - the existing submission spec becomes the infrastructure that makes tokenization possible. Receipts become token IDs. Rehydration cards become constraint documentation.
`.trim();

const Visualization = ({ engine, width, height, isRunning }: { engine: RosettaEngine, width: number, height: number, isRunning: boolean }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    let animationFrameId: number;

    const render = () => {
      // 1. Simulate Physics
      engine.simulatePhysics(width, height);
      
      // 2. Clear
      ctx.fillStyle = '#0a0a0a';
      ctx.fillRect(0, 0, width, height);
      
      // 3. Draw Edges
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
      ctx.lineWidth = 1;
      engine.adjacency.forEach((neighbors, id) => {
        const a = engine.nodes.get(id);
        if (!a) return;
        neighbors.forEach(nid => {
          const b = engine.nodes.get(nid);
          if (!b) return;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        });
      });
      
      // 4. Draw Path History
      if (engine.visitedPath.length > 1) {
        ctx.beginPath();
        const start = engine.nodes.get(engine.visitedPath[0]);
        if(start) ctx.moveTo(start.x, start.y);
        
        for (let i = 1; i < engine.visitedPath.length; i++) {
          const n = engine.nodes.get(engine.visitedPath[i]);
          if(n) ctx.lineTo(n.x, n.y);
        }
        ctx.strokeStyle = 'rgba(0, 255, 255, 0.3)';
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // 5. Draw Nodes
      engine.nodes.forEach(node => {
        // Hue mapping:
        // Novelty -> Blue (z), Loopiness -> Green (y), Risk/Boundary -> Red (x)
        const r = Math.min(255, Math.floor(node.hue.boundaryPressure * 255 + node.hue.risk * 200));
        const g = Math.min(255, Math.floor(node.hue.loopiness * 255));
        const b = Math.min(255, Math.floor(node.hue.novelty * 255));
        
        const isCurrent = node.id === engine.currentNodeId;
        const size = isCurrent ? 6 : 3 + Math.log(node.visitCount + 1);
        
        ctx.fillStyle = `rgb(${r}, ${g + 50}, ${b + 50})`;
        if (isCurrent) ctx.fillStyle = '#fff';
        
        ctx.beginPath();
        ctx.arc(node.x, node.y, size, 0, Math.PI * 2);
        ctx.fill();
        
        if (isCurrent) {
            // Glow
            ctx.shadowColor = '#0ff';
            ctx.shadowBlur = 15;
            ctx.stroke();
            ctx.shadowBlur = 0;
        }
      });
      
      animationFrameId = requestAnimationFrame(render);
    };
    
    render();
    return () => cancelAnimationFrame(animationFrameId);
  }, [engine, width, height]); // Re-bind if dimensions change

  return <canvas ref={canvasRef} width={width} height={height} className="border border-slate-700 rounded-lg bg-black/50" />;
};

const TokenCard: React.FC<{ token: StereogramToken }> = ({ token }) => (
  <div className="bg-slate-900 border border-emerald-500/30 p-3 rounded mb-2 text-xs font-mono shadow-[0_0_10px_rgba(16,185,129,0.1)]">
    <div className="flex justify-between items-center text-emerald-400 mb-1">
      <span className="font-bold">STEREOGRAM TOKEN</span>
      <span className="opacity-50">{new Date(token.timestamp).toLocaleTimeString()}</span>
    </div>
    <div className="text-slate-400 text-[10px] break-all mb-2">{token.id}</div>
    <div className="grid grid-cols-2 gap-2 text-[10px] text-slate-300">
      <div>ΔEntropy: {token.snapEvent.deltaEntropy.toFixed(4)}</div>
      <div>Resonance: {token.snapEvent.eigenResonance.toFixed(3)}</div>
      <div>Turns: {token.snapEvent.dragonSequence.length}</div>
      <div>Gain: {token.snapEvent.evalGain.toFixed(3)}</div>
    </div>
  </div>
);

const App = () => {
  const [engine] = useState(() => new RosettaEngine());
  const [corpus, setCorpus] = useState(DEFAULT_CORPUS);
  const [isRunning, setIsRunning] = useState(false);
  const [stepCount, setStepCount] = useState(0);
  const [tokens, setTokens] = useState<StereogramToken[]>([]);
  const [lastEntropy, setLastEntropy] = useState(0);
  const [flash, setFlash] = useState(false);

  const ingest = useCallback(() => {
    engine.ingest(corpus);
    setStepCount(0);
    setTokens([]);
    setLastEntropy(0);
    setIsRunning(false);
  }, [corpus, engine]);

  // Initial ingest
  useEffect(() => {
    ingest();
  }, []);

  // Run Loop
  useEffect(() => {
    let interval: number;
    if (isRunning) {
      interval = window.setInterval(() => {
        const result = engine.step();
        setStepCount(prev => prev + 1);
        setLastEntropy(engine.entropyHistory[engine.entropyHistory.length - 1] || 0);
        
        if (result.snapped && result.token) {
          setTokens(prev => [result.token!, ...prev]);
          setFlash(true);
          setTimeout(() => setFlash(false), 200);
        }
      }, 50); // Speed
    }
    return () => clearInterval(interval);
  }, [isRunning, engine]);

  return (
    <div className="min-h-screen bg-[#050505] text-slate-200 font-mono flex flex-col p-4 overflow-hidden relative">
      {flash && <div className="absolute inset-0 bg-emerald-500/10 z-50 pointer-events-none transition-opacity duration-200" />}
      
      <header className="mb-4 border-b border-slate-800 pb-2 flex justify-between items-end">
        <div>
          <h1 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-cyan-500 tracking-tighter">
            ROSETTA PHASE 2
          </h1>
          <p className="text-xs text-slate-500">Corpus-as-Terrain // Dragon Curve Navigation // Hue Memory</p>
        </div>
        <div className="flex gap-4 text-xs">
           <div className="flex flex-col items-end">
             <span className="text-slate-500">NODES</span>
             <span className="text-emerald-400 font-bold">{engine.nodes.size}</span>
           </div>
           <div className="flex flex-col items-end">
             <span className="text-slate-500">STEPS</span>
             <span className="text-cyan-400 font-bold">{stepCount}</span>
           </div>
        </div>
      </header>

      <div className="flex-1 grid grid-cols-12 gap-4 h-full overflow-hidden">
        
        {/* LEFT COLUMN: Controls */}
        <div className="col-span-3 flex flex-col gap-4 overflow-y-auto pr-2">
          <div className="bg-slate-900/50 p-3 rounded border border-slate-800">
            <label className="block text-xs text-slate-500 mb-2 uppercase tracking-widest">Corpus Ingestion</label>
            <textarea 
              value={corpus}
              onChange={(e) => setCorpus(e.target.value)}
              className="w-full h-32 bg-black border border-slate-700 rounded p-2 text-[10px] text-slate-300 focus:border-emerald-500 outline-none resize-none font-mono"
            />
            <button 
              onClick={ingest}
              className="mt-2 w-full py-1 bg-slate-800 hover:bg-slate-700 text-xs uppercase tracking-widest text-slate-400 border border-slate-700 rounded transition-colors"
            >
              Re-Ingest Terrain
            </button>
          </div>

          <div className="bg-slate-900/50 p-3 rounded border border-slate-800">
            <label className="block text-xs text-slate-500 mb-2 uppercase tracking-widest">Engine Control</label>
            <div className="flex gap-2">
              <button 
                onClick={() => setIsRunning(!isRunning)}
                className={`flex-1 py-2 text-xs font-bold rounded uppercase tracking-widest transition-all ${
                  isRunning 
                  ? 'bg-red-500/20 text-red-400 border border-red-500/50 shadow-[0_0_10px_rgba(239,68,68,0.2)]' 
                  : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/50 shadow-[0_0_10px_rgba(16,185,129,0.2)]'
                }`}
              >
                {isRunning ? 'Halt' : 'Traverse'}
              </button>
            </div>
          </div>

          <div className="bg-slate-900/50 p-3 rounded border border-slate-800 flex-1">
             <label className="block text-xs text-slate-500 mb-2 uppercase tracking-widest">Current Node</label>
             <div className="text-[10px] text-slate-400 leading-relaxed border-l-2 border-cyan-500 pl-2">
               {engine.currentNodeId ? engine.nodes.get(engine.currentNodeId)?.content : "SYSTEM IDLE"}
             </div>
             
             <div className="mt-4 grid grid-cols-2 gap-2">
                <div className="text-[10px] text-slate-500">ENTROPY</div>
                <div className="text-[10px] text-right font-mono text-cyan-300">{lastEntropy.toFixed(4)}</div>
             </div>
          </div>
        </div>

        {/* MIDDLE COLUMN: Vis */}
        <div className="col-span-6 flex flex-col relative h-[600px] min-h-0">
          <Visualization engine={engine} width={800} height={600} isRunning={isRunning} />
          
          <div className="absolute bottom-4 left-4 pointer-events-none">
             <div className="text-[10px] text-slate-500 mb-1">HUE MEMORY STATE</div>
             <div className="flex gap-1">
                <div className="w-2 h-2 bg-red-500 rounded-full"></div> <span className="text-[9px] text-slate-600 mr-2">RISK</span>
                <div className="w-2 h-2 bg-green-500 rounded-full"></div> <span className="text-[9px] text-slate-600 mr-2">LOOP</span>
                <div className="w-2 h-2 bg-blue-500 rounded-full"></div> <span className="text-[9px] text-slate-600">NOVELTY</span>
             </div>
          </div>
        </div>

        {/* RIGHT COLUMN: Tokens */}
        <div className="col-span-3 flex flex-col h-full min-h-0">
          <div className="mb-2 flex justify-between items-center">
            <span className="text-xs text-slate-500 uppercase tracking-widest">Token Feed</span>
            <span className="text-[10px] bg-emerald-900 text-emerald-300 px-1 rounded">{tokens.length} MINTED</span>
          </div>
          <div className="flex-1 overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-slate-700">
            {tokens.length === 0 ? (
              <div className="text-center text-slate-700 text-xs py-10 italic">
                Awaiting snap detection...
              </div>
            ) : (
              tokens.map(t => <TokenCard key={t.id} token={t} />)
            )}
          </div>
        </div>

      </div>
    </div>
  );
};

const rootElement = document.getElementById('root');
if (rootElement) {
  const root = createRoot(rootElement);
  root.render(<App />);
}
