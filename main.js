/*
 * Evolution Simulator
 *
 * This script implements a simple evolutionary simulation of creatures
 * moving around a grid world. Each creature has a sparsely connected
 * neural network encoded by a genome. The genome describes weighted
 * connections between inputs, internal neurons and outputs. Creatures
 * evolve over generations through selection and mutation. A brain
 * visualization is rendered for any selected creature. Population
 * statistics are plotted over time.
 */

// Size of the square grid world (worldSize x worldSize)
const worldSize = 128;
// Pixel size of each grid cell on the canvas
const cellSize = 4;
// Number of creatures in the population
const populationSize = 60;
// Number of simulation steps per generation
let generationLength = 250;
// Maximum number of internal neurons allowed in any creature
const maxInternalNeurons = 4;
// List of input names (sensors). Order is important.
const inputNames = [
  'x',
  'y',
  'age',
  'distToEast',
  'oscillator',
  'random'
];
// List of output names (actions). Order is important.
const outputNames = [
  'moveNorth',
  'moveSouth',
  'moveEast',
  'moveWest',
  'moveRandom'
];

// Simulation state variables
let creatures = [];
let stepCount = 0;
let generation = 0;
let avgFitness = 0;
let statsHistory = [];
let selectedCreature = null;
let worldCtx, overlayCtx, chartCtx;
let worldCanvas, overlayCanvas, chartCanvas;
let mutationRate = 0.05;
// Number of simulation ticks executed per animation frame; controlled by speed slider
let ticksPerFrame = 1;

// Reproduction zones: 3x3 grid of booleans; true = reproduction allowed (green), false = blocked (red)
let zones = Array.from({ length: 3 }, () => Array(3).fill(true));
// Whether the user is currently editing zones (click toggles zone instead of selecting creature)
let editingZones = false;
// Track dark mode state
let darkMode = false;

// Abbreviations and bullet descriptions for nodes
const inputAbbrev = ['X', 'Y', 'A', 'D', 'Osc', 'Rnd'];
const inputBullets = [
  ['Normalized X position (0..1)'],
  ['Normalized Y position (0..1)'],
  ['Age fraction of generation'],
  ['Distance to east wall'],
  ['Sinusoidal oscillator'],
  ['Random value [-1..1]']
];
const outputAbbrev = ['N', 'S', 'E', 'W', 'Rnd'];
const outputBullets = [
  ['Move north'],
  ['Move south'],
  ['Move east'],
  ['Move west'],
  ['Move randomly']
];

/**
 * Creature class representing an individual in the simulation.
 * Each creature has a position, age, genome, and colour. The genome is
 * an array of connection objects with source and target types/indices
 * and a weight. A colour is assigned for visual distinction.
 */
class Creature {
  constructor(id, genome) {
    this.id = id;
    // Random starting position within world bounds
    this.x = Math.floor(Math.random() * worldSize);
    this.y = Math.floor(Math.random() * worldSize);
    this.age = 0;
    this.genome = genome || Creature.randomGenome();
    // Derive internal neuron count from genome
    this.internalCount = Creature.deriveInternalCount(this.genome);
    // Assign a random colour using HSL; hue derived from id for variety
    const hue = (id * 137) % 360;
    this.color = `hsl(${hue}, 65%, 55%)`;
  }

  /**
   * Compute the number of internal neurons used by this genome. The
   * internalCount is the highest internal index referenced plus one,
   * but never greater than maxInternalNeurons.
   */
  static deriveInternalCount(genome) {
    let maxIndex = -1;
    for (const gene of genome) {
      if (gene.toType === 'internal') {
        maxIndex = Math.max(maxIndex, gene.toIndex);
      }
      if (gene.fromType === 'internal') {
        maxIndex = Math.max(maxIndex, gene.fromIndex);
      }
    }
    return Math.min(maxInternalNeurons, maxIndex + 1);
  }

  /**
   * Generate a random genome for a new creature. A genome is a list
   * of connections, each connecting an input or internal neuron to
   * either an internal neuron or an output neuron. We ensure at
   * least a few genes exist to give the brain something to work with.
   */
  static randomGenome() {
    const genome = [];
    // Choose a random number of internal neurons between 0 and maxInternalNeurons
    const internalCount = Math.floor(Math.random() * (maxInternalNeurons + 1));
    // Determine a random number of genes; at least 3 for functionality
    const geneCount = 3 + Math.floor(Math.random() * 5);
    for (let i = 0; i < geneCount; i++) {
      const { fromType, fromIndex } = Creature.randomSource(internalCount);
      const { toType, toIndex } = Creature.randomTarget(internalCount);
      // Prevent self-loop on same internal neuron to avoid trivial loops
      if (fromType === toType && fromIndex === toIndex) {
        // Skip this gene and create a new one instead
        i--;
        continue;
      }
      const weight = Creature.randomWeight();
      genome.push({ fromType, fromIndex, toType, toIndex, weight });
    }
    return genome;
  }

  /** Generate a random weight for a gene. We bias towards small weights. */
  static randomWeight() {
    // Random value between -1 and 1 with triangular distribution
    return (Math.random() - Math.random()) * 2;
  }

  /**
   * Pick a random source neuron for a new gene. Sources can be inputs,
   * internal neurons or built-in oscillators/random. Because the
   * oscillator and random sensors are treated as inputs, they are
   * implicitly part of the inputNames list.
   */
  static randomSource(internalCount) {
    // Choose between input or internal uniformly
    if (Math.random() < 0.7 || internalCount === 0) {
      // Input: choose index from inputNames array
      const index = Math.floor(Math.random() * inputNames.length);
      return { fromType: 'input', fromIndex: index };
    } else {
      // Internal
      const index = Math.floor(Math.random() * internalCount);
      return { fromType: 'internal', fromIndex: index };
    }
  }

  /**
   * Pick a random target neuron for a new gene. Targets can be internal
   * neurons (to build processing) or output neurons (to perform actions).
   */
  static randomTarget(internalCount) {
    // Choose between internal or output uniformly
    if (Math.random() < 0.5 && internalCount > 0) {
      const index = Math.floor(Math.random() * internalCount);
      return { toType: 'internal', toIndex: index };
    } else {
      const index = Math.floor(Math.random() * outputNames.length);
      return { toType: 'output', toIndex: index };
    }
  }

  /**
   * Step the creature forward by one simulation tick. We update its age,
   * compute its brain outputs and move accordingly. Movement is
   * constrained to the world boundaries.
   */
  step() {
    this.age++;
    // Compute input values (normalised to roughly [0,1] or [-1,1])
    const inputs = [];
    inputs[0] = this.x / (worldSize - 1); // x normalized
    inputs[1] = this.y / (worldSize - 1); // y normalized
    inputs[2] = this.age / generationLength; // age normalized to generation length
    inputs[3] = (worldSize - 1 - this.x) / (worldSize - 1); // distance to east wall
    // Oscillator produces a sinusoidal wave with period ~30 steps
    inputs[4] = Math.sin((this.age / 15) * 2 * Math.PI);
    // Random sensor produces random value each step
    inputs[5] = Math.random() * 2 - 1; // range [-1,1]
    // Prepare internal neuron activations
    const internal = new Array(this.internalCount).fill(0);
    // Compute internal neuron values by summing weighted inputs
    // Only a single pass; recurrent connections may behave like feed-forward due to this simple update
    for (let i = 0; i < this.internalCount; i++) {
      let sum = 0;
      for (const gene of this.genome) {
        if (gene.toType === 'internal' && gene.toIndex === i) {
          let val = 0;
          if (gene.fromType === 'input') {
            val = inputs[gene.fromIndex];
          } else if (gene.fromType === 'internal') {
            // Use previous internal values (no recurrence within same tick)
            val = internal[gene.fromIndex];
          }
          sum += val * gene.weight;
        }
      }
      // Activation function: hyperbolic tangent
      internal[i] = Math.tanh(sum);
    }
    // Compute output neuron activations
    const outputs = new Array(outputNames.length).fill(0);
    for (let j = 0; j < outputNames.length; j++) {
      let sum = 0;
      for (const gene of this.genome) {
        if (gene.toType === 'output' && gene.toIndex === j) {
          let val = 0;
          if (gene.fromType === 'input') {
            val = inputs[gene.fromIndex];
          } else if (gene.fromType === 'internal') {
            val = internal[gene.fromIndex];
          }
          sum += val * gene.weight;
        }
      }
      outputs[j] = Math.tanh(sum);
    }
    // Choose action probabilistically based on positive activations
    // Compute probabilities from positive outputs
    const probs = outputs.map(v => (v > 0 ? v : 0));
    const sum = probs.reduce((a, b) => a + b, 0);
    if (sum > 0.001) {
      // Normalize probabilities
      let r = Math.random() * sum;
      let idx = 0;
      while (r > probs[idx] && idx < probs.length - 1) {
        r -= probs[idx];
        idx++;
      }
      this.performAction(idx);
    } else {
      // No positive drive: take no action
    }
  }

  /**
   * Perform an action given by its index in outputNames. Movement is
   * restricted to the world boundaries.
   */
  performAction(actionIndex) {
    switch (outputNames[actionIndex]) {
      case 'moveNorth':
        if (this.y > 0) this.y--;
        break;
      case 'moveSouth':
        if (this.y < worldSize - 1) this.y++;
        break;
      case 'moveEast':
        if (this.x < worldSize - 1) this.x++;
        break;
      case 'moveWest':
        if (this.x > 0) this.x--;
        break;
      case 'moveRandom':
        const dirs = [
          [0, -1],
          [0, 1],
          [1, 0],
          [-1, 0]
        ];
        const [dx, dy] = dirs[Math.floor(Math.random() * dirs.length)];
        const newX = this.x + dx;
        const newY = this.y + dy;
        if (newX >= 0 && newX < worldSize) this.x = newX;
        if (newY >= 0 && newY < worldSize) this.y = newY;
        break;
    }
  }

  /**
   * Calculate fitness for this creature. Fitness is defined here as the
   * creature's x-coordinate normalised to [0,1], rewarding those who
   * reach the eastern side of the world. Custom fitness functions can
   * replace this for different behaviours.
   */
  getFitness() {
    /**
     * Calculate a simple fitness for this creature.  In earlier versions of the
     * simulator fitness was defined by how far east a creature travelled.  That
     * encouraged a strong eastward bias and survivors were ranked by their
     * x‑coordinate.  To remove this directional bias we instead define
     * fitness purely by whether the creature finishes the generation in a
     * reproduction‑allowed zone.  A creature in a green zone receives a
     * fitness of 1, while those in red zones receive 0.  The value is used
     * only for reporting and is no longer used for survivor selection, which
     * happens independently in processGenerationEnd().
     */
    const zoneX = Math.floor(this.x / (worldSize / 3));
    const zoneY = Math.floor(this.y / (worldSize / 3));
    return zones[zoneY][zoneX] ? 1 : 0;
  }
}

/**
 * Mutate a genome to produce a new, slightly altered genome. Mutation
 * may adjust weights, rewire connections, add genes or remove genes.
 * Mutations occur at a rate defined by the global mutationRate.
 * @param {Array} genome Source genome to mutate
 * @returns {Array} Mutated copy of the genome
 */
function mutateGenome(genome) {
    // Deep copy genome
    const newGenome = genome.map(g => ({ ...g }));
    // Mutate existing genes
    for (const gene of newGenome) {
        if (Math.random() < mutationRate) {
            // Adjust weight by small random amount
            gene.weight += (Math.random() - 0.5)* 0.5;
            // Clamp weight to a reasonable range
            gene.weight = Math.max(Math.min(gene.weight, 3), -3);
        }
        if (Math.random() < mutationRate) {
            // Rewire either the source or target
            const internalCount = Creature.deriveInternalCount(newGenome);
            if (Math.random() < 0.5) {
                const { fromType, fromIndex } = Creature.randomSource(internalCount);
                gene.fromType = fromType;
                gene.fromIndex = fromIndex;
            } else {
                const { toType, toIndex } = Creature.randomTarget(internalCount);
                gene.toType = toType;
                gene.toIndex = toIndex;
            }
        }
    }
    // With some probability, add a new gene
    if (Math.random() < mutationRate && newGenome.length < 12) {
        const internalCount = Creature.deriveInternalCount(newGenome);
        const { fromType, fromIndex } = Creature.randomSource(internalCount);
        const { toType, toIndex } = Creature.randomTarget(internalCount);
        // Avoid self-loop
        if (!(fromType === toType && fromIndex === toIndex)) {
            newGenome.push({ fromType, fromIndex, toType, toIndex, weight: Creature.randomWeight() });
        }
    }
    // With some probability, remove a gene (but keep at least one)
    if (Math.random() < mutationRate && newGenome.length > 1) {
        const idx = Math.floor(Math.random() * newGenome.length);
        newGenome.splice(idx, 1);
    }
    return newGenome;
}

/**
 * Initialise the simulation: set up canvases, UI controls, population,
 * and begin the animation loop.
 */
function setup() {
    // Grab DOM elements
    worldCanvas = document.getElementById('worldCanvas');
    overlayCanvas = document.getElementById('overlayCanvas');
    chartCanvas = document.getElementById('chartCanvas');
    const genSpan = document.getElementById('generation');
    const stepSpan = document.getElementById('step');
    const genLenSpan = document.getElementById('generationLength');
    const avgFitnessSpan = document.getElementById('avgFitness');
    const mutationLabel = document.getElementById('mutationRateLabel');
    const mutationSlider = document.getElementById('mutationRateSlider');
    const toggleZonesBtn = document.getElementById('toggleZonesBtn');
    const toggleDarkBtn = document.getElementById('toggleDarkBtn');
    const speedSlider = document.getElementById('speedSlider');
    const speedLabel = document.getElementById('speedLabel');
    const skipGenBtn = document.getElementById('skipGenBtn');
    // Set canvas sizes
    worldCanvas.width = worldSize * cellSize;
    worldCanvas.height = worldSize * cellSize;
    overlayCanvas.width = worldCanvas.width;
    overlayCanvas.height = worldCanvas.height;
    chartCanvas.width = document.getElementById('chartContainer').clientWidth;
    chartCanvas.height = document.getElementById('chartContainer').clientHeight;
    worldCtx = worldCanvas.getContext('2d');
    overlayCtx = overlayCanvas.getContext('2d');
    chartCtx = chartCanvas.getContext('2d');
    // Display generation length
    genLenSpan.textContent = generationLength;
    // Initialise population
    creatures = [];
    for (let i = 0; i < populationSize; i++) {
        creatures.push(new Creature(i));
    }
    // Setup mutation slider
    mutationSlider.value = mutationRate;
    mutationLabel.textContent = mutationRate.toFixed(2);
    mutationSlider.addEventListener('input', () => {
        mutationRate = parseFloat(mutationSlider.value);
        mutationLabel.textContent = mutationRate.toFixed(2);
    });
    // Toggle editing reproduction zones
    toggleZonesBtn.addEventListener('click', () => {
        editingZones = !editingZones;
        // Update button appearance
        toggleZonesBtn.classList.toggle('active', editingZones);
    });
    // Toggle dark mode
    toggleDarkBtn.addEventListener('click', () => {
        darkMode = !darkMode;
        document.body.classList.toggle('dark-mode', darkMode);
        drawWorld();
        drawOverlay();
        drawChart();
    });
    // Canvas click handler for selecting creatures
    overlayCanvas.addEventListener('click', event => {
        const rect = overlayCanvas.getBoundingClientRect();
        const clickX = event.clientX - rect.left;
        const clickY = event.clientY - rect.top;
        const gridX = Math.floor(clickX / cellSize);
        const gridY = Math.floor(clickY / cellSize);
        // Determine which 3x3 zone was clicked
        const zoneX = Math.floor((gridX / worldSize) * 3);
        const zoneY = Math.floor((gridY / worldSize) * 3);
        if (editingZones) {
            // Toggle reproduction allowed for this zone
            zones[zoneY][zoneX] = !zones[zoneY][zoneX];
            drawWorld();
            drawOverlay();
            return;
        }
        // If not editing zones, select creature near clicked cell
        let found = null;
        let minDist = Infinity;
        // Increase click radius for easier selection
        const maxDistSq = 25; // within 5 cells
        for (const creature of creatures) {
            const dx = creature.x - gridX;
            const dy = creature.y - gridY;
            const dist = dx * dx + dy * dy;
            if (dist <= maxDistSq && dist < minDist) {
                found = creature;
                minDist = dist;
            }
        }
        selectedCreature = found;
        drawBrainGraph(selectedCreature);
    });

    // Setup speed slider
    speedSlider.value = ticksPerFrame;
    speedLabel.textContent = ticksPerFrame + 'x';
    speedSlider.addEventListener('input', () => {
        ticksPerFrame = parseInt(speedSlider.value, 10);
        speedLabel.textContent = ticksPerFrame + 'x';
    });
    // Skip generation button
    skipGenBtn.addEventListener('click', () => {
        skipOneGeneration();
        // Redraw world and chart after skip
        drawWorld();
        drawOverlay();
        drawChart();
 });
    // Start the simulation loop
    function animate() {
        // Perform multiple simulation ticks per frame according to speed
        for (let t = 0; t < ticksPerFrame; t++) {
            if (stepCount >= generationLength) {
                processGenerationEnd();
            }
            for (const creature of creatures) {
                creature.step();
            }
            stepCount++;
        }
        // Draw world and overlay once per frame
        drawWorld();
        drawOverlay();
        // Update UI
        genSpan.textContent = generation;
        stepSpan.textContent = stepCount;
        avgFitnessSpan.textContent = avgFitness.toFixed(3);
        requestAnimationFrame(animate);
    }
    animate();
}

/**
 * Draw the world and all creatures onto the world canvas. The world
 * background is white. Creatures are drawn as small coloured squares.
 */
function drawWorld() {
    // Read colours from CSS variables via computed styles
    const styles = getComputedStyle(document.body);
    const bg = styles.getPropertyValue('--panel-bg');
    const gridColor = styles.getPropertyValue('--grid-line-color');
    const zoneAllow = styles.getPropertyValue('--zone-allow');
    const zoneBlock = styles.getPropertyValue('--zone-block');
    // Clear world with background colour
    worldCtx.fillStyle = bg;
    worldCtx.fillRect(0, 0, worldCanvas.width, worldCanvas.height);
    // Draw reproduction zones overlay (3x3)
    const zoneWidth = worldCanvas.width / 3;
    const zoneHeight = worldCanvas.height / 3;
    for (let y = 0; y < 3; y++) {
        for (let x = 0; x < 3; x++) {
            worldCtx.fillStyle = zones[y][x] ? zoneAllow : zoneBlock;
            worldCtx.fillRect(x * zoneWidth, y * zoneHeight, zoneWidth, zoneHeight);
        }
    }
    // Draw zone boundaries
    worldCtx.strokeStyle = styles.getPropertyValue('--border-color');
    worldCtx.lineWidth = 1;
    for (let i = 1; i < 3; i++) {
        // vertical boundary
        worldCtx.beginPath();
        worldCtx.moveTo(i * zoneWidth, 0);
        worldCtx.lineTo(i * zoneWidth, worldCanvas.height);
        worldCtx.stroke();
        // horizontal boundary
        worldCtx.beginPath();
        worldCtx.moveTo(0, i * zoneHeight);
        worldCtx.lineTo(worldCanvas.width, i * zoneHeight);
        worldCtx.stroke();
    }
    // Draw grid lines lightly every 32 cells for orientation
    worldCtx.strokeStyle = gridColor;
    worldCtx.lineWidth = 1;
    for (let i = 0; i <= worldSize; i += 32) {
        const pos = i * cellSize;
        // vertical
        worldCtx.beginPath();
        worldCtx.moveTo(pos, 0);
        worldCtx.lineTo(pos, worldCanvas.height);
        worldCtx.stroke();
        // horizontal
        worldCtx.beginPath();
        worldCtx.moveTo(0, pos);
        worldCtx.lineTo(worldCanvas.width, pos);
        worldCtx.stroke();
    }
    // Draw creatures as small squares
    for (const creature of creatures) {
        worldCtx.fillStyle = creature.color;
        worldCtx.fillRect(creature.x * cellSize, creature.y * cellSize, cellSize, cellSize);
    }
}

/**
 * Draw overlays such as selected creature highlight on the overlay canvas.
 */
function drawOverlay() {
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    if (selectedCreature) {
        const styles = getComputedStyle(document.body);
        const highlight = styles.getPropertyValue('--highlight-color');
        overlayCtx.strokeStyle = highlight;
        overlayCtx.lineWidth = 2;
        overlayCtx.strokeRect(
            selectedCreature.x * cellSize - 2,
            selectedCreature.y * cellSize - 2,
            cellSize + 4,
            cellSize + 4
        );
    }
}

/**
 * End-of-generation processing. Compute fitness of all creatures, update
 * stats history, produce next generation by selecting top performers
 * and reproducing them with mutation.
 */
function processGenerationEnd() {
    // Determine which creatures end the generation in reproduction-allowed zones
    const allowedCandidates = creatures.filter(c => {
        const zoneX = Math.floor(c.x / (worldSize / 3));
        const zoneY = Math.floor(c.y / (worldSize / 3));
        return zones[zoneY][zoneX];
    });
    // Update average fitness as fraction of creatures in allowed zones (adaptation metric)
    avgFitness = allowedCandidates.length / populationSize;
    statsHistory.push(avgFitness);
    drawChart();
    // Determine survivors: random sample from allowed candidates or, if none, from all creatures
    const pool = allowedCandidates.length > 0 ? allowedCandidates : creatures;
    const survivors = [];
    const survivorCount = Math.max(1, Math.floor(populationSize * 0.2));
    // Simple random sampling without replacement
    const indices = Array.from({ length: pool.length }, (_, i) => i);
    for (let i = 0; i < survivorCount && indices.length > 0; i++) {
        const r = Math.floor(Math.random() * indices.length);
        const idx = indices.splice(r, 1)[0];
        survivors.push(pool[idx]);
    }
    // Form new population by duplicating survivors and mutating their genomes
    const newCreatures = [];
    let idCounter = 0;
    while (newCreatures.length < populationSize) {
        for (const parent of survivors) {
            if (newCreatures.length >= populationSize) break;
            const childGenome = mutateGenome(parent.genome);
            const child = new Creature(idCounter++, childGenome);
            newCreatures.push(child);
        }
    }
    creatures = newCreatures;
    // Reset state for new generation
    generation++;
    stepCount = 0;
    selectedCreature = null;
    drawBrainGraph(null);
}

/**
 * Render the brain graph of a creature onto the SVG element. If no
 * creature is provided, clear the SVG. Nodes are positioned in
 * vertical layers: inputs on the left, internal neurons in the centre
 * and outputs on the right. Edges are drawn with colours to reflect
 * the sign of the weight.
 * @param {Creatur
