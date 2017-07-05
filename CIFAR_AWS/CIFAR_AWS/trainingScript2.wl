(* ::Package:: *)

(* ::Title:: *)
(*CIFAR-10 AWS Script*)


(* ::Subsection:: *)
(*Data import*)


dir := Directory[]


CloudConnect["laurensorensen@college.harvard.edu", "Chumble1"];


cifar = ResourceObject["CIFAR-10"];
train = ResourceData[cifar, "TrainingData"];
test = ResourceData[cifar, "TestData"];


trainRand = RandomSample[train];
trainingSet = trainRand[[ ;; 45000]];
validationSet = trainRand[[45001 ;; ]];


classes = Union@Values[train];


checkpointsPath0 = 
	FileNameJoin[{Directory[],"checkpoints/net0"}];
checkpointsPath = 
	Table[FileNameJoin[{Directory[],"checkpoints/net"}]<>IntegerString[i], {i, 1, 10}];


(* Defining a log to record weights, gradients and batch loss at training checkpoints. *)
logFile0 = OpenWrite[];
logFiles = Table[OpenWrite[], 10];


cmPath0 = 
	FileNameJoin[{Directory[],"classifier_measurements/net0"}];
cmPath = 
	Table[FileNameJoin[{Directory[],"classifier_measurements/net"}]<>IntegerString[i], {i, 1, 10}];


logPath0 = 
	FileNameJoin[{Directory[],"training_logs/net0"}];
logPath = 
	Table[FileNameJoin[{Directory[],"training_logs/net"}]<>IntegerString[i], {i, 1, 10}];


lenet = NetChain[{
			ConvolutionLayer[20, 5], 
			Ramp, 
			PoolingLayer[2, 2], 
			ConvolutionLayer[50, 5], 
			Ramp, 
			PoolingLayer[2, 2], 
			FlattenLayer[], 
			500, 
			Ramp, 
			10, SoftmaxLayer[]
		}, 
		"Output" -> NetDecoder[{"Class", classes}], 
		"Input" -> NetEncoder[{"Image", {32, 32}}] 
	]


weightLayers = {1,4,8,10}


appendToLog[n_] := PutAppend[ 
					<|"Weights" -> #Weights,
					"Gradients" -> #Gradients, 
					"BatchLoss" -> #BatchLoss,
					"ValidationLoss" -> #ValidationLoss|>, 
				logFiles[[n]] ]&


net = NetInitialize[
		Fold[
			NetReplacePart[#,{#2,"Weights"} -> Automatic]&,
			lenet,
			weightLayers
		]
	];


(* Freeze 2nd linear layer. *)
net6 = 
	NetTrain[
		net,
		trainingSet,
		ValidationSet -> Scaled[0.1],
		MaxTrainingRounds -> 150,
		BatchSize -> 100,
		TrainingProgressFunction -> {appendToLog[6], "Interval" -> Quantity[5, "Rounds"]},
		TrainingProgressCheckpointing-> {"Directory", checkpointsPath[[6]], "Interval" -> Quantity[5, "Rounds"]},
		TargetDevice -> "GPU",
		LearningRateMultipliers -> 
			{10 -> 0}
];


Export[ FileNameJoin@{checkpointsPath[[6]], "net6.wlnet"}, net6];


Export[ FileNameJoin@{cmPath[[6]], "net6.mx"}, ClassifierMeasurements[ net6, validationSet ]];


(* Reading the checkpointing log. *)
log6 = ReadList[logFiles[[6]]];
(* Saving the net and log to disk. *)
Export[ FileNameJoin@{logPath[[6]], "net6.mx"}, log6];


(* Freeze all but 1st conv. layer. *)
net7 = 
	NetTrain[
		net,
		trainingSet,
		ValidationSet -> validationSet,
		MaxTrainingRounds -> 150,
		BatchSize -> 100,
		TrainingProgressFunction -> {appendToLog[7], "Interval" -> Quantity[5, "Rounds"]},
		TrainingProgressCheckpointing-> {"Directory", checkpointsPath[[7]], "Interval" -> Quantity[5, "Rounds"]},
		TargetDevice -> "GPU",
		LearningRateMultipliers -> 
			{4 -> 0, 8 -> 0, 10 -> 0}
];


Export[ FileNameJoin@{checkpointsPath[[7]], "net7.wlnet"}, net7];


Export[ FileNameJoin@{cmPath[[7]], "net7.mx"}, ClassifierMeasurements[ net7, validationSet ]];


(* Reading the checkpointing log. *)
log7 = ReadList[logFiles[[7]]];
(* Saving the net and log to disk. *)
Export[ FileNameJoin@{logPath[[7]], "net7.mx"}, log7];


(* Freeze all but 2nd conv. layer *)
net8 = 
	NetTrain[
		net,
		trainingSet,
		ValidationSet -> validationSet,
		MaxTrainingRounds -> 150,
		BatchSize -> 100,
		TrainingProgressFunction -> {appendToLog[8], "Interval" -> Quantity[5, "Rounds"]},
		TrainingProgressCheckpointing-> {"Directory", checkpointsPath[[8]], "Interval" -> Quantity[5, "Rounds"]},
		TargetDevice -> "GPU",
		LearningRateMultipliers -> 
			{1 -> 0, 8 -> 0, 10 -> 0}
];


Export[ FileNameJoin@{checkpointsPath[[8]], "net8.wlnet"}, net8];


Export[ FileNameJoin@{cmPath[[8]], "net8.mx"}, ClassifierMeasurements[ net8, validationSet ]];


(* Reading the checkpointing log. *)
log8 = ReadList[logFiles[[8]]];
(* Saving the net and log to disk. *)
Export[ FileNameJoin@{logPath[[8]], "net8.mx"}, log8];


(* Freeze all but 1st linear layer *)
net9 = 
	NetTrain[
		net,
		trainingSet,
		ValidationSet -> validationSet,
		MaxTrainingRounds -> 150,
		BatchSize -> 100,
		TrainingProgressFunction -> {appendToLog[9], "Interval" -> Quantity[5, "Rounds"]},
		TrainingProgressCheckpointing-> {"Directory", checkpointsPath[[9]], "Interval" -> Quantity[5, "Rounds"]},
		TargetDevice -> "GPU",
		LearningRateMultipliers -> 
			{1 -> 0, 4 -> 0, 10 -> 0}
];


Export[ FileNameJoin@{checkpointsPath[[9]], "net9.wlnet"}, net9];


Export[ FileNameJoin@{cmPath[[9]], "net9.mx"}, ClassifierMeasurements[ net9, validationSet ]];


(* Reading the checkpointing log. *)
log9 = ReadList[logFiles[[9]]];
(* Saving the net and log to disk. *)
Export[ FileNameJoin@{logPath[[9]], "net9.mx"}, log9];


(* Freeze all but 2nd linear layer *)
net10 = 
	NetTrain[
		net,
		trainingSet,
		ValidationSet -> validationSet,
		MaxTrainingRounds -> 150,
		BatchSize -> 100,
		TrainingProgressFunction -> {appendToLog[10], "Interval" -> Quantity[5, "Rounds"]},
		TrainingProgressCheckpointing-> {"Directory", checkpointsPath[[10]], "Interval" -> Quantity[5, "Rounds"]},
		TargetDevice -> "GPU",
		LearningRateMultipliers -> 
			{1 -> 0, 4 -> 0, 8 -> 0}
];


Export[ FileNameJoin@{checkpointsPath[[10]], "net10.wlnet"}, net10];


Export[ FileNameJoin@{cmPath[[10]], "net10.mx"}, ClassifierMeasurements[ net10, validationSet ]];


(* Reading the checkpointing log. *)
log10 = ReadList[logFiles[[10]]];
(* Saving the net and log to disk. *)
Export[ FileNameJoin@{logPath[[10]], "net10.mx"}, log10];
