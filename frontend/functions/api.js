import { lambdaClient } from "@/clients/aws";
import { AWS_LAMBDA_NAME } from "@/constants";
import { InvokeCommand } from "@aws-sdk/client-lambda";

export const getDateRange = async () => {
    const payload = { task: "get_metadata" };
    const command = new InvokeCommand({
        FunctionName: AWS_LAMBDA_NAME,
        Payload: JSON.stringify(payload),
    });
    const { Payload } = await lambdaClient.send(command);
    const result = Buffer.from(Payload).toString();
    const response = JSON.parse(result);
    return response;
};

export const forecastCounts = async (args) => {
    const payload = { task: "predict", arguments: args };
    const command = new InvokeCommand({
        FunctionName: AWS_LAMBDA_NAME,
        Payload: JSON.stringify(payload),
    });
    console.log("Infering...");
    const { Payload } = await lambdaClient.send(command);
    const result = Buffer.from(Payload).toString();
    const response = JSON.parse(result);
    return response;
};

export const getStats = async (args) => {
    const payload = { task: "get_stats", arguments: args };
    const command = new InvokeCommand({
        FunctionName: AWS_LAMBDA_NAME,
        Payload: JSON.stringify(payload),
    });
    const { Payload } = await lambdaClient.send(command);
    const result = Buffer.from(Payload).toString();
    const response = JSON.parse(result);
    return response;
};
