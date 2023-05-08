import React from "react";
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
} from "chart.js";
import { Line } from "react-chartjs-2";
import { Box } from "@mui/material";
// import faker from "faker";

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
);

export const options = {
    responsive: true,
    plugins: {
        legend: {
            position: "top",
        },
        title: {
            display: true,
            text: "Forecast",
        },
    },
};

const ForecastChart = ({ sx, data }) => {
    if (Object.keys(data).length === 0) return <></>;
    else {
        const processed_data = { datasets: [] };
        for (const country in data) {
            if (Object.hasOwnProperty.call(data, country)) {
                const dataset = data[country];
                processed_data.labels = dataset.date;
                processed_data.datasets.push({
                    label: country,
                    data: dataset.count,
                    borderColor:
                        "#" +
                        (((1 << 24) * Math.random()) | 0)
                            .toString(16)
                            .padStart(6, "0"),
                });
            }
        }
        return (
            <Box sx={sx}>
                <Line options={options} data={processed_data} />
            </Box>
        );
    }
};

export default ForecastChart;
