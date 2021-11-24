// Copyright (C) 2021, Mindee.

// This program is licensed under the Apache License version 2.
// See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import React from "react";
import { Box, makeStyles, Theme, Typography } from "@material-ui/core";
import { Card } from "@mindee/web-elements.ui.card";
import { ModelConfig } from "src/common/types";
import Select from "react-select";
import { DET_CONFIG, RECO_CONFIG } from "src/common/constants";

const COMPONENT_ID = "Sidebar";

const useStyles = makeStyles((theme: Theme) => ({
  wrapper: {
    height: "95vh",
  },
}));

interface Props {
  detConfig: ModelConfig;
  setDetConfig: (value: any) => void;
  recoConfig: ModelConfig;
  setRecoConfig: (value: any) => void;
}
export default function Sidebar({
  detConfig,
  setDetConfig,
  recoConfig,
  setRecoConfig,
}: Props): JSX.Element {
  const classes = useStyles();
  return (
    <Card header="Model selection" id={COMPONENT_ID} className={classes.wrapper}>
      <Box display="flex" flexDirection="column ">
        <Typography>Text detection architecture (backbone)</Typography>
        <Select
          value={detConfig}
          onChange={(value) => setDetConfig(value)}
          options={Object.values(DET_CONFIG)}
        />
      </Box>
      <Box display="flex" flexDirection="column ">
        <Typography>Text recognition architecture (backbone)</Typography>
        <Select
          value={recoConfig}
          onChange={(value) => setRecoConfig(value)}
          options={Object.values(RECO_CONFIG)}
        />
      </Box>
    </Card>
  );
}
