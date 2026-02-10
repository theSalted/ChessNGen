<script lang="ts">
	import { onMount } from 'svelte';
	import type { PageData } from './$types';

	let { data }: { data: PageData } = $props();

	let mode = $state<'opening' | 'startpos'>('opening');
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
	let opening = $state('');
	let models = $state<{id: string, name: string}[]>([]);
	let selectedModel = $state<string | null>(null);
	let autoSteps = $state(10);
	let autoPlaying = $state(false);
	let stopRequested = $state(false);
	let looping = $state(true);
	let loopTimer = $state<ReturnType<typeof setInterval> | null>(null);

	onMount(async () => {
		// Fetch available models
		try {
			const res = await fetch('/api/models');
			if (res.ok) {
				const modelData = await res.json();
				models = modelData.models;
				if (models.length > 0 && selectedModel === null) {
					selectedModel = modelData.active || models[models.length - 1].id;
				}
			}
		} catch {}

		if (data.frames.length > 0) {
			frames = data.frames;
			opening = data.opening;
			currentFrame = frames.length - 1;
			initialized = true;
			resumeLoop();
		} else {
			init();
		}
	});

	async function init() {
		loading = true;
		frames = [];
		partialHistory = [];
		currentFrame = 0;
		initialized = false;
		partialFrame = null;
		opening = '';

		try {
			const res = await fetch('/api/init', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ mode, model_id: selectedModel })
			});

			if (!res.ok) {
				const err = await res.json();
				alert(err.detail || 'Init failed');
				return;
			}

			const data = await res.json();
			frames = data.frames;
			opening = data.opening;
			currentFrame = frames.length - 1;
			initialized = true;
			resumeLoop();
		} catch (e) {
			alert(`Failed to connect to backend: ${e}`);
		} finally {
			loading = false;
		}
	}

	function step(): Promise<boolean> {
		return new Promise((resolve) => {
			if (!initialized || stepping) { resolve(false); return; }
			pauseLoop();
			stepping = true;
			partialFrame = null;
			currentFrame = frames.length - 1;
			let stepPartials: string[] = [];

			const params = new URLSearchParams({
				temperature: temperature.toString(),
				top_k: topK.toString()
			});

			const es = new EventSource(`/api/step?${params}`);

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
				if (!autoPlaying) resumeLoop();
				resolve(true);
			});

			es.onerror = () => {
				stepping = false;
				partialFrame = null;
				es.close();
				if (!autoPlaying) resumeLoop();
				resolve(false);
			};
		});
	}

	let autoRemaining = $state(0);

	async function autoPlay() {
		if (autoPlaying) return;
		autoPlaying = true;
		stopRequested = false;
		autoRemaining = autoSteps;
		for (let i = 0; i < autoSteps; i++) {
			if (stopRequested) break;
			const ok = await step();
			if (!ok) break;
			autoRemaining--;
		}
		autoPlaying = false;
		stopRequested = false;
		resumeLoop();
	}

	function stopAutoPlay() {
		stopRequested = true;
	}

	function pauseLoop() {
		if (loopTimer) clearInterval(loopTimer);
		loopTimer = null;
	}

	function resumeLoop() {
		if (!looping || loopTimer) return;
		loopTimer = setInterval(() => {
			currentFrame = (currentFrame + 1) % frames.length;
		}, 500);
	}

	function toggleLoop() {
		if (looping) {
			pauseLoop();
			looping = false;
		} else {
			looping = true;
			currentFrame = 0;
			resumeLoop();
		}
	}

	async function loadImage(b64: string): Promise<HTMLImageElement> {
		const img = new Image();
		img.src = `data:image/png;base64,${b64}`;
		await new Promise((resolve) => (img.onload = resolve));
		return img;
	}

	function grabPixels(ctx: CanvasRenderingContext2D): Uint8Array {
		return new Uint8Array(ctx.getImageData(0, 0, 256, 256).data.buffer);
	}

	async function downloadGif() {
		if (frames.length === 0) return;
		const { encode } = await import('modern-gif');

		const canvas = document.createElement('canvas');
		canvas.width = 256;
		canvas.height = 256;
		const ctx = canvas.getContext('2d')!;

		const gifFrames = [];
		for (let i = 4; i <= currentFrame; i++) {
			if (includePartials) {
				const partialIdx = i - 4;
				if (partialIdx >= 0 && partialIdx < partialHistory.length) {
					const prevImg = await loadImage(frames[i - 1]);
					for (const b64 of partialHistory[partialIdx]) {
						const partialImg = await loadImage(b64);
						// Draw previous frame, then overlay partial at 50%
						ctx.globalAlpha = 1.0;
						ctx.drawImage(prevImg, 0, 0, 256, 256);
						ctx.globalAlpha = 0.5;
						ctx.drawImage(partialImg, 0, 0, 256, 256);
						ctx.globalAlpha = 1.0;
						gifFrames.push({ data: grabPixels(ctx), delay: 100 });
					}
				}
			}
			const img = await loadImage(frames[i]);
			ctx.globalAlpha = 1.0;
			ctx.drawImage(img, 0, 0, 256, 256);
			gifFrames.push({ data: grabPixels(ctx), delay: 500 });
		}

		const output = await encode({ width: 256, height: 256, frames: gifFrames as any });
		const blob = new Blob([output], { type: 'image/gif' });
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = 'chessngen.gif';
		a.click();
		URL.revokeObjectURL(url);
	}

	let openingMoves = $derived(opening ? opening.split(' ') : []);



</script>

{#snippet openingCard()}
	<div class="text-xs font-medium mb-2" style="color: #8a6a4e;">Opening</div>
	<div class="flex flex-col gap-0.5 text-sm font-mono" style="color: #f0d9b5;">
		{#each Array(Math.ceil(openingMoves.length / 2)) as _, i}
			<div class="flex gap-2">
				<span style="color: #8a6a4e;">{i + 1}.</span>
				<span class="w-10">{openingMoves[i * 2]}</span>
				<span class="w-10" style="color: #b58863;">{openingMoves[i * 2 + 1] ?? ''}</span>
			</div>
		{/each}
	</div>
{/snippet}

<svelte:head>
	<title>ChessNGen Playground</title>
</svelte:head>

<div class="min-h-screen flex flex-col items-center px-4 py-8" style="background: #000; color: #f0d9b5;">

	<!-- Board + Slider (left) + Opening card (right) -->
	<div class="relative">
		<div
			class="w-[512px] h-[512px] rounded-lg flex items-center justify-center overflow-hidden"
			style="background: #3d2e25; border: 2px solid #b58863;"
		>
			{#if loading}
				<div class="text-sm animate-pulse" style="color: #b58863;">Loading...</div>
			{:else if stepping && partialFrame && frames.length > 0}
				<img
					src={`data:image/png;base64,${frames[frames.length - 1]}`}
					alt="Previous frame"
					class="w-full h-full absolute inset-0"
					style="image-rendering: pixelated;"
				/>
				<img
					src={`data:image/png;base64,${partialFrame}`}
					alt="Generating"
					class="w-full h-full absolute inset-0 opacity-50"
					style="image-rendering: pixelated;"
				/>
			{:else if frames.length > 0}
				<img
					src={`data:image/png;base64,${frames[currentFrame]}`}
					alt="Frame {currentFrame}"
					class="w-full h-full"
					style="image-rendering: pixelated;"
				/>
			{:else}
				<div class="text-sm animate-pulse" style="color: #b58863;">Loading board...</div>
			{/if}
		</div>

		<!-- Slider (leading/left edge) -->
		{#if frames.length > 0}
			<div class="absolute top-0 -left-12 h-[512px] flex flex-col items-center">
				<div class="flex-1 flex items-center justify-center" style="width: 24px;">
					<input
						type="range"
						min={0}
						max={frames.length - 1}
						bind:value={currentFrame}
						oninput={() => { if (looping) toggleLoop(); }}
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

		<!-- Opening card (desktop: right edge) -->
		{#if opening}
			<div class="absolute top-0 -right-48 w-40 rounded-lg p-3 hidden lg:block" style="background: #1a1210; border: 1px solid #3d2e25;">
				{@render openingCard()}
			</div>
		{/if}
	</div>

	<!-- Next / Download / Reset -->
	<div class="mt-4 flex gap-3 w-[512px] items-center">
		<button
			onclick={() => step()}
			disabled={stepping || !initialized}
			class="flex-1 h-12 font-medium rounded-full text-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
			style="background: #f0d9b5; color: #302420;"
			onmouseenter={(e) => e.currentTarget.style.background = '#e5ccaa'}
			onmouseleave={(e) => e.currentTarget.style.background = '#f0d9b5'}
		>
			{#if stepping && !autoPlaying}
				<svg class="animate-spin h-5 w-5 mx-auto" style="color: #302420;" viewBox="0 0 24 24" fill="none">
					<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
					<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
				</svg>
			{:else}
				Next
			{/if}
		</button>

		<!-- Loop playback (circle) -->
		<button
			onclick={toggleLoop}
			disabled={frames.length <= 1 || stepping}
			class="w-12 h-12 rounded-full flex items-center justify-center disabled:opacity-50 transition-colors shrink-0"
			style="background: {looping ? '#f0d9b5' : '#b58863'}; color: #302420;"
			onmouseenter={(e) => e.currentTarget.style.background = looping ? '#e5ccaa' : '#a07753'}
			onmouseleave={(e) => e.currentTarget.style.background = looping ? '#f0d9b5' : '#b58863'}
			title={looping ? 'Stop loop' : 'Loop playback'}
		>
			<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/></svg>
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
	</div>

	<!-- Auto-play -->
	<div class="mt-4 w-[512px]">
		<div class="text-xs font-medium mb-2" style="color: #8a6a4e;">Auto-play</div>
		<div class="flex gap-2 items-center">
			<div class="flex items-center h-10 rounded-full shrink-0 overflow-hidden" style="background: #3d2e25; border: 1px solid #b58863;">
				<button
					onclick={() => { if (autoSteps > 1) autoSteps--; }}
					disabled={autoPlaying || autoSteps <= 1}
					class="w-8 h-full flex items-center justify-center disabled:opacity-30 transition-colors"
					style="color: #b58863;"
					onmouseenter={(e) => e.currentTarget.style.background = '#4a3a2f'}
					onmouseleave={(e) => e.currentTarget.style.background = 'transparent'}
				>
					<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><polyline points="15,18 9,12 15,6"/></svg>
				</button>
				<span class="w-8 text-center text-sm font-medium tabular-nums" style="color: #f0d9b5;">{autoPlaying ? autoRemaining : autoSteps}</span>
				<button
					onclick={() => { if (autoSteps < 500) autoSteps++; }}
					disabled={autoPlaying}
					class="w-8 h-full flex items-center justify-center disabled:opacity-30 transition-colors"
					style="color: #b58863;"
					onmouseenter={(e) => e.currentTarget.style.background = '#4a3a2f'}
					onmouseleave={(e) => e.currentTarget.style.background = 'transparent'}
				>
					<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><polyline points="9,6 15,12 9,18"/></svg>
				</button>
			</div>
			{#if autoPlaying}
				<button
					onclick={stopAutoPlay}
					class="flex-1 h-10 rounded-full flex items-center justify-center transition-colors"
					style="background: #c44; color: #fff;"
					onmouseenter={(e) => e.currentTarget.style.background = '#a33'}
					onmouseleave={(e) => e.currentTarget.style.background = '#c44'}
					title="Stop"
				>
					<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="1"/></svg>
				</button>
			{:else}
				<button
					onclick={autoPlay}
					disabled={stepping || !initialized}
					class="flex-1 h-10 rounded-full flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
					style="background: #f0d9b5; color: #302420;"
					onmouseenter={(e) => e.currentTarget.style.background = '#e5ccaa'}
					onmouseleave={(e) => e.currentTarget.style.background = '#f0d9b5'}
					title="Auto-play"
				>
					<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><polygon points="6,4 20,12 6,20"/></svg>
				</button>
			{/if}
		</div>
	</div>

	<!-- Model selector (only shown when multiple models available) -->
	{#if models.length > 1}
		<div class="mt-4 w-[512px]">
			<div class="text-xs font-medium mb-2" style="color: #8a6a4e;">Model</div>
			<select
				bind:value={selectedModel}
				onchange={() => init()}
				disabled={loading || stepping}
				class="w-full rounded-full px-4 py-2 text-sm focus:outline-none focus:ring-1 disabled:opacity-50"
				style="background: #3d2e25; color: #f0d9b5; border: 1px solid #b58863;"
			>
				{#each models as model}
					<option value={model.id}>{model.name}</option>
				{/each}
			</select>
		</div>
	{/if}

	<!-- Opening -->
	<div class="mt-4 w-[512px]">
		<div class="text-xs font-medium mb-2" style="color: #8a6a4e;">Opening</div>
		<div class="flex gap-2">
			<button
				class="flex-1 px-3 py-1.5 rounded-full text-sm transition-colors"
				style="background: {mode === 'opening' ? '#f0d9b5' : '#1a1210'}; color: {mode === 'opening' ? '#302420' : '#8a6a4e'}; border: 1px solid {mode === 'opening' ? '#f0d9b5' : '#3d2e25'};"
				onclick={() => { mode = 'opening'; init(); }}
				disabled={loading || stepping}
			>
				Random Opening
			</button>
			<button
				class="flex-1 px-3 py-1.5 rounded-full text-sm transition-colors"
				style="background: {mode === 'startpos' ? '#f0d9b5' : '#1a1210'}; color: {mode === 'startpos' ? '#302420' : '#8a6a4e'}; border: 1px solid {mode === 'startpos' ? '#f0d9b5' : '#3d2e25'};"
				onclick={() => { mode = 'startpos'; init(); }}
				disabled={loading || stepping}
			>
				Standard Opening
			</button>
		</div>
	</div>

	<!-- Opening card (mobile: below buttons) -->
	{#if opening}
		<div class="mt-4 w-[512px] rounded-lg p-3 lg:hidden" style="background: #1a1210; border: 1px solid #3d2e25;">
			{@render openingCard()}
		</div>
	{/if}

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
