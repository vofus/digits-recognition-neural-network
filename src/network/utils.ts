export function getRandomInt(min: number, max: number): number {
	return Math.floor(Math.random() * (max - min)) + min;
}

export function getFileName(fileName: string, ext: string = ".json"): string {
	const pattern = new RegExp(`(\\${ext})$`);

	return `${fileName}${pattern.test(fileName) ? "" : ext}`;
}