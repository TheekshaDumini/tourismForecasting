import Head from "next/head";
import { Inter } from "next/font/google";
import styles from "@/styles/Home.module.css";
import { useEffect, useState } from "react";
import { getDateRange } from "@/functions/api";
import CountPredictor from "@/containers/countPredictor/CountPredictor";
import { Typography } from "@mui/material";
import TourismStats from "@/containers/tourismStats/TourismStats";

const inter = Inter({ subsets: ["latin"] });

export default function Home() {
    const [metaData, setMetaData] = useState();
    const updateMetaData = async () => {
        getDateRange()
            .then((response) => {
                if (response.status === 200) {
                    console.log(response);
                    setMetaData(response.body);
                } else {
                    console.error(response);
                    setMetaData(null);
                }
            })
            .catch((err) => {
                console.error(err);
                setMetaData(null);
            });
    };
    useEffect(() => {
        updateMetaData();
    }, []);
    return (
        <>
            <Head>
                <title>Tourism Forecaster</title>
                <meta
                    name="description"
                    content="Forecast tourism trends in Sri Lanka"
                />
                <meta
                    name="viewport"
                    content="width=device-width, initial-scale=1"
                />
                <link rel="icon" href="/favicon.ico" />
            </Head>
            <main className={styles.main}>
                <Typography variant="h4">Tourism Forecaster</Typography>
                <CountPredictor metaData={metaData} />
                <TourismStats metaData={metaData} />
            </main>
        </>
    );
}
