import { BACKEND_URL } from '$env/static/private';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async () => {
	const res = await fetch(`${BACKEND_URL}/api/init`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ mode: 'opening' })
	});

	if (!res.ok) {
		return { frames: [], opening: '' };
	}

	const data = await res.json();
	return { frames: data.frames as string[], opening: data.opening as string };
};
