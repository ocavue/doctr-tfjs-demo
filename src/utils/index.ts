// Copyright (C) 2021, Mindee.

// This program is licensed under the Apache License version 2.
// See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import cv from "@techstark/opencv-js";
import {
  argMax,
  browser,
  concat,
  GraphModel,
  loadGraphModel,
  PixelData,
  Rank,
  scalar,
  softmax,
  squeeze,
  Tensor,
  unstack,
} from "@tensorflow/tfjs";
import { Layer } from "konva/lib/Layer";
import randomColor from "randomcolor";
import { MutableRefObject } from "react";
import { Stage } from "react-mindee-js";
import {
  DET_MEAN,
  DET_STD,
  REC_MEAN,
  REC_STD,
  VOCAB,
} from "src/common/constants";
import { ModelConfig } from "src/common/types";
import { chunk } from "underscore";

export const loadRecognitionModel = async ({
  recognitionModel,
  recoConfig,
}: {
  recognitionModel: MutableRefObject<GraphModel | null>;
  recoConfig: ModelConfig;
}) => {
  try {
    recognitionModel.current = await loadGraphModel(recoConfig.path);
  } catch (error) {
    console.log(error);
  }
};

export const loadDetectionModel = async ({
  detectionModel,
  detConfig,
}: {
  detectionModel: MutableRefObject<GraphModel | null>;
  detConfig: ModelConfig;
}) => {
  try {
    detectionModel.current = await loadGraphModel(detConfig.path);
  } catch (error) {
    console.log(error);
  }
};

export const getImageTensorForRecognitionModel = (
  crops: Array<ImageData>,
  size: [number, number]
) => {
  const list = crops.map((imageObject) => {
    let h = imageObject.height;
    let w = imageObject.width;
    let resize_target: any;
    let padding_target: any;
    let aspect_ratio = size[1] / size[0];
    if (aspect_ratio * h > w) {
      resize_target = [size[0], Math.round((size[0] * w) / h)];
      padding_target = [
        [0, 0],
        [0, size[1] - Math.round((size[0] * w) / h)],
        [0, 0],
      ];
    } else {
      resize_target = [Math.round((size[1] * h) / w), size[1]];
      padding_target = [
        [0, size[0] - Math.round((size[1] * h) / w)],
        [0, 0],
        [0, 0],
      ];
    }
    return browser
      .fromPixels(imageObject)
      .resizeNearestNeighbor(resize_target)
      .pad(padding_target, 0)
      .toFloat()
      .expandDims();
  });
  const tensor = concat(list);
  let mean = scalar(255 * REC_MEAN);
  let std = scalar(255 * REC_STD);
  return tensor.sub(mean).div(std);
};

export const getImageTensorForDetectionModel = (
  imageData: ImageData,
  size: [number, number]
): Tensor<Rank> => {
  let tensor = browser
    .fromPixels(imageData)
    .resizeNearestNeighbor(size)
    .toFloat();
  let mean = scalar(255 * DET_MEAN);
  let std = scalar(255 * DET_STD);
  return tensor.sub(mean).div(std).expandDims();
};

export const extractWords = async ({
  recognitionModel,
  crops,
  size,
}: {
  recognitionModel: GraphModel | null;
  crops: ImageData[];
  size: [number, number];
}) => {
  console.log("extracting words", {
    recognitionModel,
    crops,
    size,
  });

  // 将 ImageData 转换为 PixelData
  // const pixelDataList = crops.map((crop): PixelData => {
  //   return {
  //     width: crop.width,
  //     height: crop.height,
  //     data: crop.data,
  //   };
  // });
let   pixelDataList = crops

  const chunks = chunk(pixelDataList, 32);
  return Promise.all(
    chunks.map(
      (chunk) =>
        new Promise(async (resolve) => {
          const words = await extractWordsFromCrop({
            recognitionModel,
            crops: chunk,
            size,
          });
          const collection = words?.map((word, index) => ({
            ...chunk[index],
            words: word ? [word] : [],
          }));
          resolve(collection);
        })
    )
  );
};

export const dataURItoBlob = (dataURI: string) => {
  let byteString;
  const splitDataURL = dataURI.split(",");
  if (splitDataURL[0].indexOf("base64") >= 0) {
    // atob decodes base64 data
    byteString = atob(splitDataURL[1]);
  } else {
    byteString = decodeURI(dataURI.split(",")[1]);
  }

  const mimeString = splitDataURL[0].split(":")[1].split(";")[0];

  // write the bytes of the string to a typed array
  const ia = new Uint8Array(byteString.length);
  for (let i = 0; i < byteString.length; i++) {
    ia[i] = byteString.charCodeAt(i);
  }

  return new Blob([ia], { type: mimeString });
};

export const getCrops = (
  imageData: ImageData,
  polygons: AnnotationShape[],
  size: [number, number]
): ImageData[] => {
  const canvas = new OffscreenCanvas(size[1], size[0]);
  const ctx = canvas.getContext("2d");

  if (!ctx) {
    throw new Error("Cannot get 2D context from OffscreenCanvas");
  }

  // 将原始图像数据绘制到 OffscreenCanvas
  ctx.putImageData(imageData, 0, 0);

  const crops: ImageData[] = polygons.map((polygon) => {
    const [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] = polygon.coordinates;

    // 计算裁剪区域的边界矩形
    const minX = Math.min(x1, x2, x3, x4) * size[1];
    const minY = Math.min(y1, y2, y3, y4) * size[0];
    const maxX = Math.max(x1, x2, x3, x4) * size[1];
    const maxY = Math.max(y1, y2, y3, y4) * size[0];

    const width = maxX - minX;
    const height = maxY - minY;

    // 获取裁剪区域的 ImageData
    const cropImageData = ctx.getImageData(minX, minY, width, height);

    return cropImageData;
  });

  return crops;
};

export const extractWordsFromCrop = async ({
  recognitionModel,
  crops,
  size,
}: {
  recognitionModel: GraphModel | null;
  crops: Array<ImageData>;
  size: [number, number];
}) => {
  if (!recognitionModel) {
    return;
  }

  // for (const crop of crops) {
  //   console.log("crop", crop);
  //   document.body.appendChild(crop);
  // }

  let tensor = getImageTensorForRecognitionModel(crops, size);
  let predictions = await recognitionModel.executeAsync(tensor);

  //  @ts-ignore
  // @ts-ignore
  let probabilities = softmax(predictions, -1);
  let bestPath = unstack(argMax(probabilities, -1), 0);
  let blank = 126;
  var words = [];
  for (const sequence of bestPath) {
    let collapsed = "";
    let added = false;
    const values = sequence.dataSync();
    const arr = Array.from(values);
    for (const k of arr) {
      if (k === blank) {
        added = false;
      } else if (k !== blank && added === false) {
        collapsed += VOCAB[k];
        added = true;
      }
    }
    words.push(collapsed);
  }
  return words;
};

export interface HeatMap {
  pixelData: Uint8ClampedArray;
  width: number;
  height: number;
}

export const getHeatMapFromImage = async ({
  detectionModel,
  imageData,
  size,
}: {
  detectionModel: GraphModel | null;
  imageData: ImageData;
  size: [number, number];
}): Promise<HeatMap | undefined> => {
  if (!detectionModel) {
    return;
  }
  let tensor = getImageTensorForDetectionModel(imageData, size);
  let result = detectionModel.execute(tensor);
  let squeezed = squeeze<Tensor<Rank.R3>>(
    Array.isArray(result) ? result[0] : result
  );
  const pixelData = await browser.toPixels(squeezed);
  return { pixelData, width: squeezed.shape[0], height: squeezed.shape[1] };
};

function clamp(number: number, size: number) {
  return Math.max(0, Math.min(number, size));
}

export interface AnnotationShape {
  id: number;
  coordinates: number[][];
}

export const transformBoundingBox = (
  contour: cv.Rect,
  id: number,
  size: [number, number]
): AnnotationShape => {
  let offset =
    (contour.width * contour.height * 1.8) /
    (2 * (contour.width + contour.height));
  const x1 = clamp(contour.x - offset, size[1]) - 1;
  const x2 = clamp(x1 + contour.width + 2 * offset, size[1]) - 1;
  const y1 = clamp(contour.y - offset, size[0]) - 1;
  const y2 = clamp(y1 + contour.height + 2 * offset, size[0]) - 1;
  return {
    id,
    // config: {
    //   stroke: randomColor(),
    // },
    coordinates: [
      [x1 / size[1], y1 / size[0]],
      [x2 / size[1], y1 / size[0]],
      [x2 / size[1], y2 / size[0]],
      [x1 / size[1], y2 / size[0]],
    ],
  };
};

export const extractBoundingBoxesFromHeatmap = (
  heatMap: HeatMap,
  size: [number, number]
): AnnotationShape[] => {
  // debugger;
  const imageData = new ImageData(
    new Uint8ClampedArray(heatMap.pixelData),
    heatMap.width,
    heatMap.height
  );
  let src = cv.matFromImageData(imageData);

  cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
  cv.threshold(src, src, 77, 255, cv.THRESH_BINARY);
  cv.morphologyEx(src, src, cv.MORPH_OPEN, cv.Mat.ones(2, 2, cv.CV_8U));
  let contours = new cv.MatVector();
  let hierarchy = new cv.Mat();
  // You can try more different parameters
  cv.findContours(
    src,
    contours,
    hierarchy,
    cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_SIMPLE
  );
  // draw contours with random Scalar
  const boundingBoxes: AnnotationShape[] = [];
  for (let i = 0; i < contours.size(); ++i) {
    const contourBoundingBox = cv.boundingRect(contours.get(i));
    if (contourBoundingBox.width > 2 && contourBoundingBox.height > 2) {
      boundingBoxes.unshift(transformBoundingBox(contourBoundingBox, i, size));
    }
  }
  src.delete();
  contours.delete();
  hierarchy.delete();
  return boundingBoxes;
};

// 新增函数，用于将图像转换为 ImageData
export const getImageData = (image: HTMLImageElement): Promise<ImageData> => {
  return new Promise((resolve, reject) => {
    const canvas = document.createElement("canvas");
    canvas.width = image.width;
    canvas.height = image.height;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      reject("Cannot get 2D context");
      return;
    }
    ctx.drawImage(image, 0, 0);
    const imageData = ctx.getImageData(0, 0, image.width, image.height);
    resolve(imageData);
  });
};
