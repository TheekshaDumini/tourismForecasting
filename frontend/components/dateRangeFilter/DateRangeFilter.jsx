import Box from "@mui/material/Box";
import Slider from "@mui/material/Slider";
import { useEffect, useState } from "react";

const monthDiff = (d1, d2) => {
    let months;
    months = (d2.getFullYear() - d1.getFullYear()) * 12;
    months -= d1.getMonth();
    months += d2.getMonth();
    return months <= 0 ? 0 : months;
};
const addMonths = (months, date) => {
    const newDate = new Date(date);
    const d = date.getDate();
    newDate.setMonth(newDate.getMonth() + months);
    if (newDate.getDate() != d) {
        newDate.setDate(0);
    }
    return newDate;
};

const DateRangeFilter = ({ dateRange, setRange, sx }) => {
    const availableStart =
        dateRange &&
        new Date(
            `${dateRange.available.start.year}-${dateRange.available.start.month}`
        );
    const availableEnd =
        dateRange &&
        new Date(
            `${dateRange.available.end.year}-${dateRange.available.end.month}`
        );
    const predictableStart =
        dateRange &&
        new Date(
            `${dateRange.available.start.year}-${dateRange.available.start.month}`
        );
    const predictableEnd =
        dateRange &&
        new Date(
            `${dateRange.predictable.end.year}-${dateRange.predictable.end.month}`
        );
    const numMonths = dateRange && monthDiff(availableStart, predictableEnd);
    const num2Date = (val, startDate = availableStart) => {
        if (startDate) {
            const date = addMonths(val, startDate);
            const monthNames = [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ];
            const year = date.getFullYear();
            const month = monthNames[date.getMonth()];
            return `${year} ${month}`;
        } else {
            return val;
        }
    };

    const [value, setValue] = useState([0, numMonths]);
    const updateRange = (lowerLim, upperLim) => {
        const stringLowerLim = num2Date(lowerLim);
        const year = stringLowerLim.split(" ")[0];
        const month = stringLowerLim.split(" ")[1];
        const n_months = upperLim - lowerLim;
        const newRange = { year, month, n_months };
        setRange(newRange);
    };
    const handleChange = (event, newValue) => {
        setValue(newValue);
        updateRange(...newValue);
    };
    const marks = [
        {
            value: 0,
            label: num2Date(0),
        },
        {
            value: numMonths,
            label: num2Date(numMonths),
        },
    ];
    // if (value[1]) updateRange(...value);
    return (
        <Box sx={{ ...sx, minWidth: "50vw" }}>
            <Slider
                value={value}
                onChange={handleChange}
                valueLabelDisplay="auto"
                valueLabelFormat={(val) => num2Date(val)}
                disableSwap={true}
                max={numMonths}
                marks={marks}
            />
        </Box>
    );
};
export default DateRangeFilter;
