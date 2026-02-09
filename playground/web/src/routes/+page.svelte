<script lang="ts">
	let mode = $state<'startpos' | 'pgn'>('startpos');
	let pgn = $state('');
	let temperature = $state(0.0);
	let topK = $state(0);

	let frames = $state<string[]>([]);
	let currentFrame = $state(0);
	let loading = $state(false);
	let stepping = $state(false);
	let initialized = $state(false);
	let showAdvanced = $state(false);
	let partialFrame = $state<string | null>(null);

	const API_URL = 'http://localhost:8000';

	// Auto-init on mount
	$effect(() => {
		init();
	});

	async function init() {
		loading = true;
		frames = [];
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

		const params = new URLSearchParams({
			temperature: temperature.toString(),
			top_k: topK.toString()
		});

		const es = new EventSource(`${API_URL}/api/step?${params}`);

		es.addEventListener('partial', (e) => {
			const data = JSON.parse(e.data);
			partialFrame = data.frame;
		});

		es.addEventListener('done', (e) => {
			const data = JSON.parse(e.data);
			partialFrame = null;
			frames = [...frames, data.frame];
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

	function prevFrame() {
		if (currentFrame > 0) currentFrame--;
	}

	let displaySrc = $derived(
		stepping && partialFrame
			? `data:image/png;base64,${partialFrame}`
			: frames.length > 0
				? `data:image/png;base64,${frames[currentFrame]}`
				: null
	);
</script>

<div class="min-h-screen bg-neutral-950 text-neutral-100 flex flex-col items-center px-4 py-8">
	<h1 class="text-2xl font-bold mb-6">ChessNGen Playground</h1>

	<!-- Board -->
	<div
		class="w-[512px] h-[512px] bg-neutral-900 rounded-lg flex items-center justify-center overflow-hidden"
	>
		{#if loading}
			<div class="text-neutral-500 text-sm animate-pulse">Loading...</div>
		{:else if displaySrc}
			<img
				src={displaySrc}
				alt="Frame {currentFrame}"
				class="w-full h-full"
				style="image-rendering: pixelated;"
			/>
		{:else}
			<div class="text-neutral-500 text-sm animate-pulse">Loading board...</div>
		{/if}
	</div>

	<!-- Scrubber + frame nav -->
	{#if frames.length > 0}
		<input
			type="range"
			min={0}
			max={frames.length - 1}
			bind:value={currentFrame}
			class="w-[512px] mt-4"
		/>
		<div class="flex items-center gap-4 mt-2">
			<button
				onclick={prevFrame}
				disabled={currentFrame === 0}
				class="px-3 py-1 bg-neutral-800 rounded text-sm hover:bg-neutral-700 disabled:opacity-30"
			>
				Prev
			</button>
			<span class="text-sm text-neutral-400 tabular-nums">
				{currentFrame + 1} / {frames.length}
			</span>
		</div>
	{/if}

	<!-- Next / Reset -->
	<div class="mt-4 flex gap-3">
		<button
			onclick={step}
			disabled={stepping || !initialized}
			class="bg-white text-black font-medium px-6 py-2 rounded hover:bg-neutral-200 disabled:opacity-50 disabled:cursor-not-allowed"
		>
			{stepping ? 'Generating...' : 'Next'}
		</button>
		<button
			onclick={init}
			disabled={loading || stepping}
			class="px-4 py-2 bg-neutral-800 rounded text-sm text-neutral-300 hover:bg-neutral-700 disabled:opacity-50"
		>
			Reset
		</button>
	</div>

	<!-- Advanced -->
	<div class="mt-6 w-[512px]">
		<button
			onclick={() => (showAdvanced = !showAdvanced)}
			class="text-sm text-neutral-500 hover:text-neutral-300"
		>
			{showAdvanced ? '- Advanced' : '+ Advanced'}
		</button>

		{#if showAdvanced}
			<div class="flex flex-col gap-4 mt-3">
				<div>
					<label class="block text-sm text-neutral-400 mb-1">Bootstrap</label>
					<div class="flex gap-2">
						<button
							class="px-3 py-1.5 rounded text-sm {mode === 'startpos'
								? 'bg-white text-black'
								: 'bg-neutral-800 text-neutral-300'}"
							onclick={() => (mode = 'startpos')}
						>
							Standard Opening
						</button>
						<button
							class="px-3 py-1.5 rounded text-sm {mode === 'pgn'
								? 'bg-white text-black'
								: 'bg-neutral-800 text-neutral-300'}"
							onclick={() => (mode = 'pgn')}
						>
							From PGN
						</button>
					</div>
				</div>

				{#if mode === 'pgn'}
					<div>
						<label class="block text-sm text-neutral-400 mb-1" for="pgn">PGN</label>
						<textarea
							id="pgn"
							bind:value={pgn}
							rows={4}
							class="w-full bg-neutral-800 rounded px-3 py-2 text-sm font-mono text-neutral-100 resize-none focus:outline-none focus:ring-1 focus:ring-neutral-500"
							placeholder="1. e4 e5 2. Nf3 Nc6 ..."
						></textarea>
					</div>
				{/if}

				<div>
					<label class="block text-sm text-neutral-400 mb-1" for="temp"
						>Temperature: {temperature.toFixed(2)}</label
					>
					<input
						id="temp"
						type="range"
						min={0}
						max={1.5}
						step={0.05}
						bind:value={temperature}
						class="w-full"
					/>
				</div>

				<div>
					<label class="block text-sm text-neutral-400 mb-1" for="topk">Top-k</label>
					<input
						id="topk"
						type="number"
						min={0}
						max={1000}
						bind:value={topK}
						class="w-full bg-neutral-800 rounded px-3 py-2 text-sm text-neutral-100 focus:outline-none focus:ring-1 focus:ring-neutral-500"
					/>
				</div>
			</div>
		{/if}
	</div>
</div>
