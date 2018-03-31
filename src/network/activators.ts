import { NdArray, NjParam, sigmoid } from "numjs";

export interface ActivationStrategy {
	execute<T = number>(x: NjParam<T>): NdArray<T>;
}

abstract class BasicActivation implements ActivationStrategy {
	protected abstract activate<T = number>(x: NjParam<T>): NdArray<T>;

	execute<T = number>(x: NjParam<T>): NdArray<T> {
		return this.activate(x);
	}
}


export class Sigmoid extends BasicActivation {
	protected activate<T = number>(x: NjParam<T>): NdArray<T> {
		return sigmoid(x);
	}
}