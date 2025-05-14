import { runEval } from "./lib/eval";
import { getChosenConfig } from "./lib/utils";
import * as configs from "./configs";
const chosenConfig = getChosenConfig(Object.values(configs));
const results = await runEval(chosenConfig);

console.log(JSON.stringify(results, null, 2));
const mean = results.reduce((acc, curr) => acc + curr, 0) / results.length;
console.log(`Mean: ${mean}`);
