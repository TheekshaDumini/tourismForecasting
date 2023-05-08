import ForecastChart from "@/components/forecastChart/ForecastChart";
import CountryFilter from "@/components/countryFilter/CountryFilter";
import DateRangeFilter from "@/components/dateRangeFilter/DateRangeFilter";
import { Box, Typography } from "@mui/material";
import { LoadingButton } from "@mui/lab";
import { useState } from "react";
import styles from "./styles";
import { forecastCounts } from "@/functions/api";

const CountPredictor = ({ metaData }) => {
    const [selectedCountries, setSelectedCountries] = useState([]);
    const [dateRange, setRange] = useState({
        year: undefined,
        month: undefined,
        n_months: undefined,
    });
    const [countsData, setCountsData] = useState({});
    const [infering, setInfering] = useState(false);
    return (
        <Box sx={styles.root}>
            <Typography variant="h5">Forecast</Typography>
            <CountryFilter
                sx={styles.component}
                countries={(metaData && metaData.countries) || metaData}
                setSelectedCountries={setSelectedCountries}
            />
            <ForecastChart sx={styles.component} data={countsData} />
            <Box sx={styles.btnContainer}>
                <LoadingButton
                    variant="contained"
                    sx={styles.component}
                    onClick={() => {
                        setInfering(true);
                        forecastCounts({
                            countries: selectedCountries,
                            ...dateRange,
                        })
                            .then((res) => {
                                if (res.status === 200) setCountsData(res.body);
                                else console.error(res);
                            })
                            .catch(console.error)
                            .finally(() => setInfering(false));
                    }}
                    disabled={
                        selectedCountries.length === 0 ||
                        dateRange.year === undefined
                    }
                    loading={infering}
                >
                    Fetch
                </LoadingButton>
            </Box>
            <DateRangeFilter
                dateRange={(metaData && metaData.date_range) || metaData}
                setRange={setRange}
                sx={styles.component}
            />
        </Box>
    );
};

export default CountPredictor;
