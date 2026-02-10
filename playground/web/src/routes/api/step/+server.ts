import { BACKEND_URL } from '$env/static/private';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ url }) => {
	const temperature = url.searchParams.get('temperature') ?? '0';
	const top_k = url.searchParams.get('top_k') ?? '0';

	const params = new URLSearchParams({ temperature, top_k });
	const res = await fetch(`${BACKEND_URL}/api/step?${params}`, {
		headers: { 'ngrok-skip-browser-warning': '1' }
	});

	return new Response(res.body, {
		headers: {
			'Content-Type': 'text/event-stream',
			'Cache-Control': 'no-cache',
			Connection: 'keep-alive'
		}
	});
};
