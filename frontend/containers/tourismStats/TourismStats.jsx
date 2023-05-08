import { Box, MenuItem, Select, Typography } from "@mui/material";
import styles from "./styles";
import CountsChart from "@/components/countsChart/CountsChart";
import { useEffect, useState } from "react";
import { getStats } from "@/functions/api";

const TourismStats = ({ metaData }) => {
    // console.log(metaData);
    let years = [];
    if (metaData) {
        const startYear = metaData.date_range.available.start.year;
        const endYear = metaData.date_range.available.end.year;
        console.log(startYear, endYear);
        years = Array.from(
            { length: endYear - startYear + 1 },
            (x, i) => i + startYear
        );
    }
    const [year, setYear] = useState(2018);
    const [counts, setCounts] = useState({});
    const handleChangeYear = (e) => {
        setYear(e.target.value);
    };
    useEffect(() => {
        getStats({ year })
            .then((res) => {
                if (res.status === 200) {
                    setCounts(res.body.counts);
                } else console.error(res);
            })
            .catch(console.error);
    }, []);
    return (
        <Box sx={styles.root}>
            <Typography variant="h5" sx={styles.title}>
                Tourism Statistics
            </Typography>
            <Box sx={styles.bodyContainer}>
                <Select
                    sx={styles.select}
                    value={year}
                    label="Year"
                    onChange={handleChangeYear}
                >
                    {years.map((year) => (
                        <MenuItem key={year} value={year}>
                            {year}
                        </MenuItem>
                    ))}
                </Select>
                <Box sx={styles.chartContainer}>
                    <CountsChart counts={counts} />
                </Box>
            </Box>
        </Box>
    );
};

export default TourismStats;
