import { BACKEND_URL } from '$env/static/private';
import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

export const POST: RequestHandler = async () => {
	const res = await fetch(`${BACKEND_URL}/api/init`, {
		method: 'POST'
	});

	if (!res.ok) {
		const err = await res.json();
		return json(err, { status: res.status });
	}

	return json(await res.json());
};
