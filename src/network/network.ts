import fs from "fs";
import { promisify } from "util";
import nj, { NdArray } from "numjs";
import { getFileName, getRandomInt } from "./utils";
import { ActivationStrategy, Sigmoid } from "./activators";
import { shuffle as _shuffle } from "lodash";

const writeFile = promisify(fs.writeFile);
const readFile = promisify(fs.readFile);

export type ModelNN = { IH: NdArray, HO: NdArray, LR: number, activator: ActivationStrategy };
export type Relation = "IH" | "HO";

export interface NetworkConfig {
	inputSize: number;
	hiddenSize: number;
	outputSize: number;
	LR?: number;
	MOMENT?: number;
	useRProp?: boolean;
}

export interface INetwork {
	train(trainSet: TrainSet, count: number, activator?: ActivationStrategy): void;

	query(inputs: number[]): any;
}

export interface IModel<T> {
	getModel(): T;
}

export interface IForwardResult {
	hiddenOutputs: NdArray;
	finalOutputs: NdArray;
}

export interface IBackResult {
	deltaWeightsHO: NdArray;
	deltaWeightsIH: NdArray;
	weightsHO: NdArray;
	weightsIH: NdArray;
}

export interface ITrainItem {
	inputs: number[];
	targets: number[];
}

export type TrainSet = ITrainItem[];

export class Network implements INetwork, IModel<ModelNN> {
	private static readonly LR_DEFAULT: number = 0.3;
	private static readonly MOMENT_DEFAULT: number = 0;
	private static readonly R_PROP_PARAMS = {
		LR_MIN: 0.000001,
		LR_MAX: 50,
		LR_INC_MULT: 1.2,
		LR_DEC_MULT: 0.5
	};

	// сериализуем тренированную модель
	static async serialize(nn: Network, filePath: string) {
		const { IH, HO, LR } = nn.getModel();

		try {
			const data = {
				IH: IH.tolist(),
				HO: HO.tolist(),
				LR
			};

			const jsonData = JSON.stringify(data);

			await writeFile(getFileName(filePath), jsonData);
		} catch (err) {
			throw err;
		}
	}

	// десериализуем тренированную модель
	static async deserialize(filePath: string): Promise<Network> {
		try {
			const jsonStr: string = await readFile(getFileName(filePath), "utf8");
			const { IH, HO, LR } = JSON.parse(jsonStr);
			const weightsIH = nj.array(IH);
			const weightsHO = nj.array(HO);

			const [hiddenSize, inputSize] = weightsIH.shape;
			const [outputSize] = weightsHO.shape;
			const nn: Network = new Network(inputSize, hiddenSize, outputSize);

			nn.setModel({
				IH: weightsIH,
				HO: weightsHO,
				LR,
				activator: new Sigmoid()
			});

			return nn;
		} catch (err) {
			throw err;
		}
	}

	// GENERAL_PARAMS
	// Скорость обучения
	private LR: number = 0.3;
	// Момент
	private MOMENT: number = 0;
	// Матрица весов между входным и скрытым слоем
	private weightsIH: NdArray;
	// Матрица весов между скрытым и выходным слоем
	private weightsHO: NdArray;
	// Матрица предыдущих изменений весов между входным и скрытым слоем
	private prevDeltaWeightsIH: NdArray;
	// Матрица предыдущих изменений весов между скрытым и выходным слоем
	private prevDeltaWeightsHO: NdArray;
	// Объект-активатор (По умолчанию сигмоида)
	private activator: ActivationStrategy = new Sigmoid();

	// RPROP_PARAMS
	// Флаг использования метода обучения RProp
	private useRProp: boolean = false;
	// Матрица предыдущих скоростей обучения между входным и скрытым слоем
	private prevLerningRateIH: NdArray;
	// Матрица предыдущих скоростей обучения между скрытым и выходным слоем
	private prevLerningRateHO: NdArray;
	// Матрица предыдущих ошибок между входным и скрытым слоем
	private prevErrorsIH: NdArray;
	// Матрица предыдущих ошибок между скрытым и выходным слоем
	private prevErrorsHO: NdArray;
	// Матрица предыдущих знаков градиента между входным и скрытым слоем
	private prevSignIH: NdArray;
	// Матрица предыдущих знаков градиента между скрытым и выходным слоем
	private prevSignHO: NdArray;

	/**
	 * Getter модели сети
	 */
	getModel(): ModelNN {
		return {
			IH: this.weightsIH,
			HO: this.weightsHO,
			LR: this.LR,
			activator: this.activator
		};
	}

	/**
	 * Setter модели сети
	 */
	setModel(model: ModelNN): void {
		const { IH, HO, LR, activator } = model;

		this.weightsIH = IH;
		this.weightsHO = HO;
		this.LR = LR;
		this.activator = activator;
	}

	/**
	 * Setter функции активации
	 */
	setActivatorStrategy(activator: ActivationStrategy) {
		this.activator = activator;
	}

	constructor(config: NetworkConfig)
	constructor(
		inputSize: number,
		hiddenSize: number,
		outputSize: number,
		LR?: number,
		MOMENT?: number,
		useRProp?: boolean)
	constructor(
		configOrInputSize: number | NetworkConfig,
		hiddenSize?: number,
		outputSize?: number,
		LR: number = Network.LR_DEFAULT,
		MOMENT: number = Network.MOMENT_DEFAULT,
		useRProp: boolean = false
	) {
		if (typeof configOrInputSize === "object") {
			this.initFromConfig(configOrInputSize);
		} else if (typeof configOrInputSize === "number") {
			this.LR = LR;
			this.MOMENT = MOMENT;
			this.useRProp = useRProp;
			this.weightsIH = this.generateWeights(hiddenSize, configOrInputSize);
			this.weightsHO = this.generateWeights(outputSize, hiddenSize);
		}

		if (this.useRProp) {
			this.initRProp();
		}
	}

	/**
	 * Иницмализация сети из объекта конфигурации
	 */
	private initFromConfig(config: NetworkConfig) {
		const { inputSize, hiddenSize, outputSize, LR, MOMENT, useRProp } = config;

		this.LR = typeof LR === "number" ? LR : Network.LR_DEFAULT;
		this.MOMENT = typeof MOMENT === "number" ? MOMENT : Network.MOMENT_DEFAULT;
		this.useRProp = typeof useRProp === "boolean" ? useRProp : false;
		this.weightsIH = this.generateWeights(hiddenSize, inputSize);
		this.weightsHO = this.generateWeights(outputSize, hiddenSize);
	}

	/**
	 * Инициализация дополнительных матриц для RProp
	 */
	private initRProp() {
		// Сбросить скорость обучения и момент на значения по умолчанию для алгоритма RProp
		this.LR = 0.5;
		this.MOMENT = 1;

		this.prevErrorsIH = nj.ones(this.weightsIH.shape);
		this.prevErrorsHO = nj.ones(this.weightsHO.shape);

		this.prevLerningRateIH = nj.zeros(this.weightsIH.shape).assign(this.LR, false);
		this.prevLerningRateHO = nj.zeros(this.weightsHO.shape).assign(this.LR, false);

		this.prevSignIH = nj.zeros(this.weightsIH.shape);
		this.prevSignHO = nj.zeros(this.weightsIH.shape);
	}

	/**
	 * Тренируем сеть
	 * @param trainSet {TrainSet} тренировочная выборка
	 * @param epochs {number} количество эпох обучения
	 * @param activator {ActivationStrategy} объект-активатор
	 */
	train(trainSet: TrainSet, epochs: number, activator?: ActivationStrategy): void {
		if (Boolean(activator)) {
			this.activator = activator;
		}

		let epochCounter = epochs;
		while (epochCounter > 0) {
			// const shuffled: TrainSet = _shuffle(trainSet);
			let trainCounter = trainSet.length - 1;

			console.time(`Epoch ${epochCounter}`);
			while (trainCounter >= 0) {
				const { inputs, targets } = trainSet[trainCounter];
				this.trainStep(inputs, targets);
				trainCounter -= 1;
			}

			console.timeEnd(`Epoch ${epochCounter}`);
			epochCounter -= 1;
		}
	}

	/**
	 * Выполняем запрос к сети
	 * @param inputs {number[]} входные сигналы
	 */
	query(inputs: number[]): number[] {
		const inputMatrix = nj.array(inputs).reshape(1, inputs.length).T as NdArray;
		const { finalOutputs } = this.forwardPropagation(inputMatrix);

		return finalOutputs.tolist<number[]>().reduce((res, item) => {
			res.push(...item);

			return res;
		}, []);
	}


	/**
	 * Шаг обучения
	 * @param inputs {number[]} входные сигналы
	 * @param targets {number[]} ожидаемый результат
	 */
	private trainStep(inputs: number[], targets: number[]): void {
		const inputMatrix = nj.array(inputs, "float64").reshape(1, inputs.length).T as NdArray;
		const targetMatrix = nj.array(targets, "float64").reshape(1, targets.length).T as NdArray;

		const forwardResult = this.forwardPropagation(inputMatrix);
		const backResult = this.backPropagation(inputMatrix, targetMatrix, forwardResult);

		this.prevDeltaWeightsHO = backResult.deltaWeightsHO;
		this.prevDeltaWeightsIH = backResult.deltaWeightsIH;
		this.weightsHO = backResult.weightsHO;
		this.weightsIH = backResult.weightsIH;
	}

	/**
	 * Генерируем начаьную матрицу весов
	 * @param rows {number} Количество строк
	 * @param columns {number} Количество столбцов
	 * @returns {NdArray}
	 */
	private generateWeights(rows: number, columns: number): NdArray {
		return nj.random([rows, columns]).subtract(0.5);
	}

	/**
	 * Подсчитываем дополнительные веса
	 */
	private calcAdditionalWeights(inputs: NdArray, outputs: NdArray, errors: NdArray, type: Relation): NdArray {
		const ones = nj.ones(outputs.shape) as NdArray;
		const arg1 = errors.multiply(outputs).multiply(nj.subtract(ones, outputs));
		const arg2 = inputs.T;
		let deltaWeights: NdArray;

		if (this.useRProp) {
			const temp = nj.dot(arg1, arg2);
			const LRMatrix = this.calcLRMatrixForRProp(temp, type);

			// console.log("LR: ", LRMatrix.shape, LRMatrix);
			// console.log("TEMP: ", temp.shape, temp);

			deltaWeights = temp.multiply(LRMatrix);
		} else {
			deltaWeights = nj.dot(arg1, arg2).multiply(this.LR);
		}

		if (type === "HO") {
			return Boolean(this.prevDeltaWeightsHO) && this.MOMENT !== 0
				? deltaWeights.add(this.prevDeltaWeightsHO.multiply(this.MOMENT))
				: deltaWeights;
		}

		if (type === "IH") {
			return Boolean(this.prevDeltaWeightsIH) && this.MOMENT !== 0
				? deltaWeights.add(this.prevDeltaWeightsIH.multiply(this.MOMENT))
				: deltaWeights;
		}
	}

	/**
	 * Посчитать матрицу скоростей обучения для алгоритма RProp
	 */
	private calcLRMatrixForRProp(gradient: NdArray, type: Relation): NdArray {
		const prevSignMatrix = type === "IH" ? this.prevSignIH : this.prevSignHO;
		const prevLRMatrix = type === "IH" ? this.prevLerningRateIH : this.prevLerningRateHO;
		const LRMatrix = nj.zeros(gradient.shape);
		const signMatrix = this.getSignMatrix(gradient);
		const [rows, cols] = gradient.shape;

		for (let rowIndex = 0; rowIndex < rows; ++rowIndex) {
			for (let colIndex = 0; colIndex < cols; ++colIndex) {
				const prevSign = prevSignMatrix.get(rowIndex, colIndex);
				const currentSign = signMatrix.get(rowIndex, colIndex);
				const conditionForChange = prevSign !== currentSign;

				if (conditionForChange) {
					const value = currentSign === 0 ? 0 : currentSign < 0 ? 0.5 : 1.2;
					LRMatrix.set(rowIndex, colIndex, value);
				} else {
					const value = prevLRMatrix.get(rowIndex, colIndex);
					LRMatrix.set(rowIndex, colIndex, value);
				}
			}
		}

		if (type === "IH") {
			this.prevSignIH = signMatrix;
			this.prevLerningRateIH = LRMatrix;
		}

		if (type === "HO") {
			this.prevSignHO = signMatrix;
			this.prevLerningRateHO = LRMatrix;
		}

		return LRMatrix;
	}

	/**
	 * Получить матрицу знаков
	 */
	private getSignMatrix(matrix: NdArray): NdArray {
		const signMatrix = nj.zeros(matrix.shape);
		const [rows, cols] = matrix.shape;

		for (let rowIndex = 0; rowIndex < rows; ++rowIndex) {
			for (let colIndex = 0; colIndex < cols; ++colIndex) {
				const value = matrix.get(rowIndex, colIndex);
				const sign = value === 0 ? 0 : value < 0 ? -1 : 1;

				signMatrix.set(rowIndex, colIndex, sign);
			}
		}

		return signMatrix;
	}

	/**
	 * Прямое распространение сигнала
	 * @param inputMatrix {NdArray} Входные сигналы, приобразованные в двумерный массив
	 * @returns {IForwardResult}
	 */
	private forwardPropagation(inputMatrix: NdArray): IForwardResult {
		const hiddenInputs = this.weightsIH.dot(inputMatrix);
		const hiddenOutputs = this.activator.execute(hiddenInputs);

		const finalInputs = this.weightsHO.dot(hiddenOutputs);
		const finalOutputs = this.activator.execute(finalInputs);

		return {
			hiddenOutputs,
			finalOutputs
		};
	}


	/**
	 * Обратное распространение ошибки
	 * @param inputMatrix {NdArray} Входные сигналы, приобразованные в двумерный массив
	 * @param targetMatrix {NdArray} Ожидаемые результаты, приобразованные в двумерный массив
	 * @param result {IForwardResult} Объект с выходными сигналами на слоях после прямого прохода
	 */
	private backPropagation(inputMatrix: NdArray, targetMatrix: NdArray, result: IForwardResult): IBackResult {
		const { hiddenOutputs, finalOutputs } = result;
		const outputErrors = targetMatrix.subtract(finalOutputs);
		const hiddenErrors = this.weightsHO.T.dot(outputErrors);

		const deltaWeightsHO = this.calcAdditionalWeights(hiddenOutputs, finalOutputs, outputErrors, "HO");
		const deltaWeightsIH = this.calcAdditionalWeights(inputMatrix, hiddenOutputs, hiddenErrors, "IH");

		const weightsHO = this.weightsHO.add(deltaWeightsHO);
		const weightsIH = this.weightsIH.add(deltaWeightsIH);

		return {
			deltaWeightsHO,
			deltaWeightsIH,
			weightsHO,
			weightsIH
		};
	}
}
