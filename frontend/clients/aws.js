import {
    AWS_ACCESS_KEY_ID,
    AWS_REGION,
    AWS_SECRET_ACCESS_KEY,
} from "@/constants";
import { LambdaClient } from "@aws-sdk/client-lambda";

const lambdaClient = new LambdaClient({
    region: AWS_REGION,
    credentials: {
        accessKeyId: AWS_ACCESS_KEY_ID,
        secretAccessKey: AWS_SECRET_ACCESS_KEY,
    },
});

export { lambdaClient };
