<script lang="ts">


	let mode = $state<'startpos' | 'pgn'>('startpos');
	let pgn = $state('');
	let temperature = $state(0.0);
	let topK = $state(0);

	let frames = $state<string[]>([]);
	let partialHistory = $state<string[][]>([]);
	let currentFrame = $state(0);
	let loading = $state(false);
	let stepping = $state(false);
	let initialized = $state(false);
	let showAdvanced = $state(false);
	let partialFrame = $state<string | null>(null);
	let includePartials = $state(true);

	const API_URL = 'http://localhost:8000';

	$effect(() => {
		init();
	});

	async function init() {
		loading = true;
		frames = [];
		partialHistory = [];
		currentFrame = 0;
		initialized = false;
		partialFrame = null;

		try {
			const res = await fetch(`${API_URL}/api/init`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					mode,
					pgn: mode === 'pgn' ? pgn : ''
				})
			});

			if (!res.ok) {
				const err = await res.json();
				alert(err.detail || 'Init failed');
				return;
			}

			const data = await res.json();
			frames = data.frames;
			currentFrame = frames.length - 1;
			initialized = true;
		} catch (e) {
			alert(`Failed to connect to backend: ${e}`);
		} finally {
			loading = false;
		}
	}

	function step() {
		if (!initialized || stepping) return;
		stepping = true;
		partialFrame = null;
		let stepPartials: string[] = [];

		const params = new URLSearchParams({
			temperature: temperature.toString(),
			top_k: topK.toString()
		});

		const es = new EventSource(`${API_URL}/api/step?${params}`);

		es.addEventListener('partial', (e) => {
			const data = JSON.parse(e.data);
			partialFrame = data.frame;
			stepPartials.push(data.frame);
		});

		es.addEventListener('done', (e) => {
			const data = JSON.parse(e.data);
			partialFrame = null;
			frames = [...frames, data.frame];
			partialHistory = [...partialHistory, stepPartials];
			currentFrame = frames.length - 1;
			stepping = false;
			es.close();
		});

		es.onerror = () => {
			stepping = false;
			partialFrame = null;
			es.close();
		};
	}

	async function loadImageData(ctx: CanvasRenderingContext2D, b64: string): Promise<Uint8ClampedArray> {
		const img = new Image();
		img.src = `data:image/png;base64,${b64}`;
		await new Promise((resolve) => (img.onload = resolve));
		ctx.drawImage(img, 0, 0, 256, 256);
		return ctx.getImageData(0, 0, 256, 256).data;
	}

	async function downloadGif() {
		if (frames.length === 0) return;
		const { encode } = await import('modern-gif');

		const canvas = document.createElement('canvas');
		canvas.width = 256;
		canvas.height = 256;
		const ctx = canvas.getContext('2d')!;

		const gifFrames = [];
		// Skip first 3 context frames (duplicates of starting position)
		for (let i = 3; i < frames.length; i++) {
			// Insert partial frames before each generated frame
			if (includePartials) {
				const partialIdx = i - 4; // partialHistory[0] corresponds to frames[4]
				if (partialIdx >= 0 && partialIdx < partialHistory.length) {
					for (const b64 of partialHistory[partialIdx]) {
						const data = await loadImageData(ctx, b64);
						gifFrames.push({ data, delay: 100 });
					}
				}
			}
			const data = await loadImageData(ctx, frames[i]);
			gifFrames.push({ data, delay: 500 });
		}

		const output = await encode({ width: 256, height: 256, frames: gifFrames });
		const blob = new Blob([output], { type: 'image/gif' });
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = 'chessngen.gif';
		a.click();
		URL.revokeObjectURL(url);
	}

	let displaySrc = $derived(
		stepping && partialFrame
			? `data:image/png;base64,${partialFrame}`
			: frames.length > 0
				? `data:image/png;base64,${frames[currentFrame]}`
				: null
	);
</script>

<svelte:head>
	<title>ChessNGen Playground</title>
</svelte:head>

<div class="min-h-screen flex flex-col items-center px-4 py-8" style="background: #000; color: #f0d9b5;">

	<!-- Board + Slider -->
	<div class="relative">
		<div
			class="w-[512px] h-[512px] rounded-lg flex items-center justify-center overflow-hidden"
			style="background: #3d2e25; border: 2px solid #b58863;"
		>
			{#if loading}
				<div class="text-sm animate-pulse" style="color: #b58863;">Loading...</div>
			{:else if displaySrc}
				<img
					src={displaySrc}
					alt="Frame {currentFrame}"
					class="w-full h-full"
					style="image-rendering: pixelated;"
				/>
			{:else}
				<div class="text-sm animate-pulse" style="color: #b58863;">Loading board...</div>
			{/if}
		</div>

		{#if frames.length > 0}
			<div class="absolute top-0 -right-12 h-[512px] flex flex-col items-center">
				<div class="flex-1 flex items-center justify-center" style="width: 24px;">
					<input
						type="range"
						min={0}
						max={frames.length - 1}
						bind:value={currentFrame}
						disabled={stepping}
						class="themed-slider disabled:opacity-30"
						style="width: 512px; transform: rotate(90deg); transform-origin: center center;"
					/>
				</div>
				<span class="text-xs tabular-nums" style="color: #b58863;">
					{currentFrame + 1}/{frames.length}
				</span>
			</div>
		{/if}
	</div>

	<!-- Next / Reset / Download -->
	<div class="mt-4 flex gap-3 w-[512px] items-center">
		<!-- Next (rounded pill) -->
		<button
			onclick={step}
			disabled={stepping || !initialized}
			class="flex-1 h-12 font-medium rounded-full text-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
			style="background: #f0d9b5; color: #302420;"
			onmouseenter={(e) => e.currentTarget.style.background = '#e5ccaa'}
			onmouseleave={(e) => e.currentTarget.style.background = '#f0d9b5'}
		>
			{#if stepping}
				<svg class="animate-spin h-5 w-5 mx-auto" style="color: #302420;" viewBox="0 0 24 24" fill="none">
					<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
					<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
				</svg>
			{:else}
				Next
			{/if}
		</button>

		<!-- Reset (circle) -->
		<button
			onclick={init}
			disabled={loading || stepping}
			class="w-12 h-12 rounded-full flex items-center justify-center disabled:opacity-50 transition-colors shrink-0"
			style="background: #b58863; color: #302420;"
			onmouseenter={(e) => e.currentTarget.style.background = '#a07753'}
			onmouseleave={(e) => e.currentTarget.style.background = '#b58863'}
			title="Reset"
		>
			<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>
		</button>

		<!-- Download GIF (circle) -->
		<button
			onclick={downloadGif}
			disabled={frames.length === 0 || stepping}
			class="w-12 h-12 rounded-full flex items-center justify-center disabled:opacity-50 transition-colors shrink-0"
			style="background: #b58863; color: #302420;"
			onmouseenter={(e) => e.currentTarget.style.background = '#a07753'}
			onmouseleave={(e) => e.currentTarget.style.background = '#b58863'}
			title="Download GIF"
		>
			<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
		</button>
	</div>

	<!-- Advanced -->
	<div class="mt-6 w-[512px]">
		<button
			onclick={() => (showAdvanced = !showAdvanced)}
			class="text-sm transition-colors"
			style="color: #8a6a4e;"
			onmouseenter={(e) => e.currentTarget.style.color = '#b58863'}
			onmouseleave={(e) => e.currentTarget.style.color = '#8a6a4e'}
		>
			{showAdvanced ? '- Advanced' : '+ Advanced'}
		</button>

		{#if showAdvanced}
			<div class="flex flex-col gap-4 mt-3">
				<label class="flex items-center gap-2 text-sm cursor-pointer" style="color: #b58863;">
					<input type="checkbox" bind:checked={includePartials} class="accent-[#b58863]" />
					Include generation process in GIF
				</label>

				<div>
					<label class="block text-sm mb-1" style="color: #b58863;" for="temp"
						>Temperature: {temperature.toFixed(2)}</label
					>
					<input
						id="temp"
						type="range"
						min={0}
						max={1.5}
						step={0.05}
						bind:value={temperature}
						class="w-full accent-[#b58863]"
					/>
				</div>

				<div>
					<label class="block text-sm mb-1" style="color: #b58863;" for="topk">Top-k</label>
					<input
						id="topk"
						type="number"
						min={0}
						max={1000}
						bind:value={topK}
						class="w-full rounded px-3 py-2 text-sm focus:outline-none focus:ring-1"
						style="background: #3d2e25; color: #f0d9b5; border: 1px solid #b58863;"
					/>
				</div>
			</div>
		{/if}
	</div>
</div>

<style>
	.themed-slider {
		-webkit-appearance: none;
		appearance: none;
		height: 10px;
		border-radius: 3px;
		background: #3d2e25;
		outline: none;
	}
	.themed-slider::-webkit-slider-thumb {
		-webkit-appearance: none;
		appearance: none;
		width: 16px;
		height: 16px;
		border-radius: 50%;
		background: #f0d9b5;
		border: 2px solid #b58863;
		cursor: pointer;
	}
	.themed-slider::-webkit-slider-thumb:hover {
		background: #e5ccaa;
	}
	.themed-slider::-moz-range-thumb {
		width: 16px;
		height: 16px;
		border-radius: 50%;
		background: #f0d9b5;
		border: 2px solid #b58863;
		cursor: pointer;
	}
	.themed-slider::-moz-range-track {
		height: 10px;
		background: #3d2e25;
		border-radius: 3px;
	}
</style>
