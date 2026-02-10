import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async () => {
	return { frames: [] as string[], opening: '' };
};
