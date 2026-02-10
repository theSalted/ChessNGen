import { BACKEND_URL } from '$env/static/private';
import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async () => {
	const res = await fetch(`${BACKEND_URL}/api/models`, {
		headers: { 'ngrok-skip-browser-warning': '1' }
	});

	if (!res.ok) {
		const err = await res.json();
		return json(err, { status: res.status });
	}

	return json(await res.json());
};
