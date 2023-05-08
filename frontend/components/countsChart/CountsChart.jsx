import { Chart as ChartJS, ArcElement, Tooltip, Legend } from "chart.js";
import { Doughnut } from "react-chartjs-2";

ChartJS.register(ArcElement, Tooltip, Legend);

export const options = {
    responsive: true,
    plugins: {
        legend: {
            position: "right",
        },
        title: {
            display: false,
            text: "Counts",
        },
    },
};

const CountsChart = ({ counts }) => {
    const data = {};
    const labels = Object.keys(counts);
    data.labels = labels;
    data.datasets = [];
    const colors = [];
    for (const key in counts) {
        if (Object.hasOwnProperty.call(counts, key)) {
            const element = counts[key];
            colors.push(
                "#" +
                    (((1 << 24) * Math.random()) | 0)
                        .toString(16)
                        .padStart(6, "0")
            );
        }
    }
    const dataset = {
        label: "# Visits",
        data: Object.values(counts),
        backgroundColor: colors,
        borderColor: colors,
    };
    data.datasets.push(dataset);
    return <Doughnut options={options} data={data} />;
};
export default CountsChart;
