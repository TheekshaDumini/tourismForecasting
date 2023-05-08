import Autocomplete from "@mui/material/Autocomplete";
import TextField from "@mui/material/TextField";
import { Box } from "@mui/material";

const CountryFilter = ({ countries, setSelectedCountries, sx }) => {
    return (
        <Box sx={sx}>
            <Autocomplete
                multiple
                options={countries ? countries : []}
                noOptionsText={
                    countries === undefined
                        ? "loading..."
                        : countries === null
                        ? "unable to fetch"
                        : undefined
                }
                onChange={(event, value) => setSelectedCountries(value)}
                filterSelectedOptions
                renderInput={(params) => (
                    <TextField {...params} label="Countries" />
                )}
            />
        </Box>
    );
};

export default CountryFilter;
