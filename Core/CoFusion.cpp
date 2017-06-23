/*
 * This file is part of ElasticFusion.
 *
 * Copyright (C) 2015 Imperial College London
 *
 * The use of the code within this file and all code within files that
 * make up the software that is ElasticFusion is permitted for
 * non-commercial purposes only.  The full terms and conditions that
 * apply to the code within this file are detailed within the LICENSE.txt
 * file and at <http://www.imperial.ac.uk/dyson-robotics-lab/downloads/elastic-fusion/elastic-fusion-license/>
 * unless explicitly stated.  By downloading this file you agree to
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then
 * please email researchcontracts.engineering@imperial.ac.uk.
 *
 */

#include "CoFusion.h"

CoFusion::CoFusion(const int timeDelta, const int countThresh, const float errThresh, const float covThresh, const bool closeLoops,
                   const bool iclnuim, const bool reloc, const float photoThresh, const float initConfidenceGlobal,
                   const float initConfidenceObject, const float depthCut, const float icpThresh, const bool fastOdom,
                   const float fernThresh, const bool so3, const bool frameToFrameRGB, const unsigned modelSpawnOffset,
                   const Model::MatchingType matchingType, const std::string& exportDirectory, const bool exportSegmentationResults)
    : modelMatchingType(matchingType),
      newModelListeners(0),
      inactiveModelListeners(0),
      modelToModel(Resolution::getInstance().width(), Resolution::getInstance().height(), Intrinsics::getInstance().cx(),
                   Intrinsics::getInstance().cy(), Intrinsics::getInstance().fx(), Intrinsics::getInstance().fy()),
      ferns(500, depthCut * 1000, photoThresh),
      tick(1),
      timeDelta(timeDelta),
      icpCountThresh(countThresh),
      icpErrThresh(errThresh),
      covThresh(covThresh),
      deforms(0),
      fernDeforms(0),
      consSample(20),
      resize(Resolution::getInstance().width(), Resolution::getInstance().height(), Resolution::getInstance().width() / consSample,
             Resolution::getInstance().height() / consSample),
      imageBuff(Resolution::getInstance().rows() / consSample, Resolution::getInstance().cols() / consSample),
      consBuff(Resolution::getInstance().rows() / consSample, Resolution::getInstance().cols() / consSample),
      timesBuff(Resolution::getInstance().rows() / consSample, Resolution::getInstance().cols() / consSample),
      closeLoops(closeLoops),
      iclnuim(iclnuim),
      reloc(reloc),
      lost(false),
      lastFrameRecovery(false),
      trackingCount(0),
      maxDepthProcessed(20.0f),
      rgbOnly(false),
      icpWeight(icpThresh),
      pyramid(true),
      fastOdom(fastOdom),
      initConfThresGlobal(initConfidenceGlobal),
      initConfThresObject(initConfidenceObject),
      fernThresh(fernThresh),
      so3(so3),
      frameToFrameRGB(frameToFrameRGB),
      depthCutoff(depthCut),
      modelSpawnOffset(modelSpawnOffset),
      exportSegmentation(exportSegmentationResults),
      exportDir(exportDirectory) {
  createTextures();
  createCompute();
  createFeedbackBuffers();

  labelGenerator.init(Resolution::getInstance().width(), Resolution::getInstance().height(), Segmentation::METHOD::CONNECTED_COMPONENTS);
  globalModel = std::make_shared<Model>(getNextModelID(true), initConfidenceGlobal, true, true, enablePoseLogging);
  models.push_back(globalModel);

  Stopwatch::getInstance().setCustomSignature(12431231);

  std::cout << "Initialised Multi-Object Fusion. Each model can have up to " << Model::MAX_VERTICES
            << " surfel (TEXTURE_DIMENSION: " << Model::TEXTURE_DIMENSION << "x" << Model::TEXTURE_DIMENSION << ")." << std::endl;
}

CoFusion::~CoFusion() {
  if (iclnuim) {
    savePly();
  }

  for (std::map<std::string, GPUTexture*>::iterator it = textures.begin(); it != textures.end(); ++it) {
    delete it->second;
  }

  textures.clear();

  for (std::map<std::string, ComputePack*>::iterator it = computePacks.begin(); it != computePacks.end(); ++it) {
    delete it->second;
  }

  computePacks.clear();

  for (std::map<std::string, FeedbackBuffer*>::iterator it = feedbackBuffers.begin(); it != feedbackBuffers.end(); ++it) {
    delete it->second;
  }

  feedbackBuffers.clear();

  cudaCheckError();
}

void CoFusion::preallocateModels(unsigned count) {
  for (unsigned i = 0; i < count; ++i)
    preallocatedModels.push_back(
        std::make_shared<Model>(getNextModelID(true), initConfThresObject, false, true, enablePoseLogging, modelMatchingType));
}

SegmentationResult CoFusion::performSegmentation(const FrameData& frame) {
  return labelGenerator.performSegmentation(models, frame, getNextModelID(), spawnOffset >= modelSpawnOffset);
}

void CoFusion::createTextures() {
  textures[GPUTexture::RGB] =
      new GPUTexture(Resolution::getInstance().width(), Resolution::getInstance().height(), GL_RGBA, GL_RGB, GL_UNSIGNED_BYTE, true, true);

  textures[GPUTexture::DEPTH_METRIC] =
      new GPUTexture(Resolution::getInstance().width(), Resolution::getInstance().height(), GL_LUMINANCE32F_ARB, GL_LUMINANCE, GL_FLOAT);

  textures[GPUTexture::DEPTH_METRIC_FILTERED] = new GPUTexture(Resolution::getInstance().width(), Resolution::getInstance().height(),
                                                               GL_LUMINANCE32F_ARB, GL_LUMINANCE, GL_FLOAT, false, true);

  textures[GPUTexture::MASK] = new GPUTexture(Resolution::getInstance().width(), Resolution::getInstance().height(),
                                              GL_R8UI,         // GL_R8, GL_R8UI, GL_R8I internal
                                              GL_RED_INTEGER,  // GL_RED, GL_RED_INTEGER // format
                                              GL_UNSIGNED_BYTE, false, true);

  // Visualisation only textures

  textures[GPUTexture::DEPTH_NORM] = new GPUTexture(Resolution::getInstance().width(), Resolution::getInstance().height(),
                                                    GL_LUMINANCE32F_ARB, GL_LUMINANCE, GL_FLOAT, true);

  textures[GPUTexture::MASK_COLOR] =
      new GPUTexture(Resolution::getInstance().width(), Resolution::getInstance().height(), GL_RGB, GL_RGB, GL_UNSIGNED_BYTE, true);
}

void CoFusion::createCompute() {
  computePacks[ComputePack::FILTER] = new ComputePack(loadProgramFromFile("empty.vert", "depth_bilateral_metric.frag", "quad.geom"),
                                                      textures[GPUTexture::DEPTH_METRIC_FILTERED]->texture);

  computePacks[ComputePack::METRIC_FILTERED] = new ComputePack(loadProgramFromFile("empty.vert", "depth_metric.frag", "quad.geom"),
                                                               textures[GPUTexture::DEPTH_METRIC_FILTERED]->texture);

  // Visualisation only

  computePacks[ComputePack::NORM_DEPTH] =
      new ComputePack(loadProgramFromFile("empty.vert", "depth_norm.frag", "quad.geom"), textures[GPUTexture::DEPTH_NORM]->texture);

  computePacks[ComputePack::COLORISE_MASKS] =
      new ComputePack(loadProgramFromFile("empty.vert", "int_to_color.frag", "quad.geom"), textures[GPUTexture::MASK_COLOR]->texture);
}

void CoFusion::createFeedbackBuffers() {
  feedbackBuffers[FeedbackBuffer::RAW] =
      new FeedbackBuffer(loadProgramGeomFromFile("vertex_feedback.vert", "vertex_feedback.geom"));  // Used to render raw depth data
  feedbackBuffers[FeedbackBuffer::FILTERED] = new FeedbackBuffer(loadProgramGeomFromFile("vertex_feedback.vert", "vertex_feedback.geom"));
}

void CoFusion::computeFeedbackBuffers() {
  TICK("feedbackBuffers");
  feedbackBuffers[FeedbackBuffer::RAW]->compute(textures[GPUTexture::RGB]->texture, textures[GPUTexture::DEPTH_METRIC]->texture, tick,
                                                maxDepthProcessed);

  feedbackBuffers[FeedbackBuffer::FILTERED]->compute(textures[GPUTexture::RGB]->texture,
                                                     textures[GPUTexture::DEPTH_METRIC_FILTERED]->texture, tick, maxDepthProcessed);
  TOCK("feedbackBuffers");
}

bool CoFusion::processFrame(const FrameData& frame, const Eigen::Matrix4f* inPose, const float weightMultiplier, const bool bootstrap) {
  assert(frame.depth.type() == CV_32FC1);
  assert(frame.rgb.type() == CV_8UC3);
  assert(frame.timestamp >= 0);

  TICK("Run");

  // Upload RGB to graphics card
  textures[GPUTexture::RGB]->texture->Upload(frame.rgb.data, GL_RGB, GL_UNSIGNED_BYTE);

  TICK("Preprocess");

  textures[GPUTexture::DEPTH_METRIC]->texture->Upload((float*)frame.depth.data, GL_LUMINANCE, GL_FLOAT);
  filterDepth();

  // if(frame.mask) {
  //    // Use ground-truth segmentation if provided (TODO: Overwritten at the moment)
  //    textures[GPUTexture::MASK]->texture->Upload(frame.mask, GL_LUMINANCE_INTEGER_EXT, GL_UNSIGNED_BYTE);
  //} else
  if (!enableMultipleModels) {
    // If the support for multiple objects is deactivated, segment everything as background (static scene).
    const long size = Resolution::getInstance().width() * Resolution::getInstance().height();
    unsigned char* data = new unsigned char[size];
    memset(data, 0, size);
    textures[GPUTexture::MASK]->texture->Upload(data, GL_LUMINANCE_INTEGER_EXT, GL_UNSIGNED_BYTE);
    delete[] data;
  }

  TOCK("Preprocess");

  // First run
  if (tick == 1) {
    computeFeedbackBuffers();
    globalModel->initialise(*feedbackBuffers[FeedbackBuffer::RAW], *feedbackBuffers[FeedbackBuffer::FILTERED]);
    globalModel->getFrameOdometry().initFirstRGB(textures[GPUTexture::RGB]);
  } else {
    bool trackingOk = true;

    // Regular execution, false if pose is provided by user
    if (bootstrap || !inPose) {
      Model::generateCUDATextures(textures[GPUTexture::DEPTH_METRIC_FILTERED], textures[GPUTexture::MASK]);

      TICK("odom");
      for (auto model : models) {
        model->performTracking(frameToFrameRGB, rgbOnly, icpWeight, pyramid, fastOdom, so3, maxDepthProcessed, textures[GPUTexture::RGB],
                               frame.timestamp, requiresFillIn(model));
      }
      TOCK("odom");

      if (bootstrap) {
        assert(inPose);
        globalModel->overridePose(globalModel->getPose() * (*inPose));
      }

      trackingOk = !reloc || globalModel->getFrameOdometry().lastICPError < 1e-04;

      if (enableMultipleModels) {
        auto getMaxDepth = [](const SegmentationResult::ModelData& data) -> float { return data.depthMean + data.depthStd * 1.2; };

        if (spawnOffset < modelSpawnOffset) spawnOffset++;

        SegmentationResult segmentationResult = performSegmentation(frame);
        textures[GPUTexture::MASK]->texture->Upload(segmentationResult.fullSegmentation.data, GL_LUMINANCE_INTEGER_EXT, GL_UNSIGNED_BYTE);

        if (exportSegmentation) {
          cv::Mat output;
          cv::threshold(segmentationResult.fullSegmentation, output, 254, 255, cv::THRESH_TOZERO_INV);
          cv::imwrite(exportDir + "Segmentation" + std::to_string(tick) + ".png", output);
          // cv::imwrite(exportDir + "RGB" + std::to_string(tick) + ".png", rgb);
        }

        // Spawn new model
        if (segmentationResult.hasNewLabel) {
          const SegmentationResult::ModelData& newModelData = segmentationResult.modelData.back();
          std::cout << "New label detected (" << newModelData.left << "," << newModelData.top << " " << newModelData.right << ","
                    << newModelData.bottom << ") - try relocating..." << std::endl;

          if (enableRedetection) {
            // [Removed code]
          }

          // New model
          std::cout << "Found new model." << std::endl;

          spawnObjectModel();
          spawnOffset = 0;

          newModel->setMaxDepth(getMaxDepth(newModelData));
        }

        // Set max-depth
        ModelList::iterator it = models.begin();
        for (unsigned i = 1; i < models.size(); i++) (*++it)->setMaxDepth(getMaxDepth(segmentationResult.modelData[i]));

        if (segmentationResult.hasNewLabel) {
          newModel->predictIndices(tick, maxDepthProcessed, timeDelta);

          newModel->fuse(tick, textures[GPUTexture::RGB], textures[GPUTexture::MASK], textures[GPUTexture::DEPTH_METRIC],
                         textures[GPUTexture::DEPTH_METRIC_FILTERED], maxDepthProcessed, 100);

          // newModel->predictIndices(tick, maxDepthProcessed, timeDelta);

          std::vector<float> test;
          newModel->clean(tick, test, timeDelta, maxDepthProcessed, false, textures[GPUTexture::DEPTH_METRIC_FILTERED],
                          textures[GPUTexture::MASK]);

          enableSpawnSubtraction = false;
          if (enableSpawnSubtraction) {
            globalModel->eraseErrorGeometry(textures[GPUTexture::DEPTH_METRIC_FILTERED]);
          }
          moveNewModelToList();
        }

        for (auto& m : segmentationResult.modelData) {  // FIXME reduce count somewhere
          if (m.superPixelCount <= 0 && (*m.modelListIterator)->incrementUnseenCount() > 0) {
            if (m.id != 0) {
              std::cout << "Lost a model." << std::endl;
              inactivateModel(m.modelListIterator);
            }
          }
        }

        // increase confidence of object-models
        it = models.begin();
        for (unsigned i = 1; i < models.size(); i++) {
          const float oldConf = (*++it)->getConfidenceThreshold();
          (*it)->setConfidenceThreshold(std::min(std::max(oldConf, segmentationResult.modelData[i].avgConfidence), 9.0f));
        }
      }

      if (reloc) {
        if (!lost) {
          Eigen::MatrixXd covariance = globalModel->getFrameOdometry().getCovariance();

          for (int i = 0; i < 6; i++) {
            if (covariance(i, i) > 1e-04) {
              trackingOk = false;
              break;
            }
          }

          if (!trackingOk) {
            trackingCount++;

            if (trackingCount > 10) {
              lost = true;
            }
          } else {
            trackingCount = 0;
          }
        } else if (lastFrameRecovery) {
          Eigen::MatrixXd covariance = globalModel->getFrameOdometry().getCovariance();

          for (int i = 0; i < 6; i++) {
            if (covariance(i, i) > 1e-04) {
              trackingOk = false;
              break;
            }
          }

          if (trackingOk) {
            lost = false;
            trackingCount = 0;
          }

          lastFrameRecovery = false;
        }
      }  // reloc

    }  // regular
    else {
      globalModel->overridePose(*inPose);
    }

    std::vector<Ferns::SurfaceConstraint> constraints;

    predict();

    Eigen::Matrix4f recoveryPose = globalModel->getPose();

    if (closeLoops) {
      lastFrameRecovery = false;

      TICK("Ferns::findFrame");
      recoveryPose = ferns.findFrame(constraints, globalModel->getPose(), globalModel->getFillInVertexTexture(),
                                     globalModel->getFillInNormalTexture(), globalModel->getFillInImageTexture(), tick, lost);
      TOCK("Ferns::findFrame");
    }

    std::vector<float> rawGraph;

    bool fernAccepted = false;

    if (closeLoops && ferns.lastClosest != -1) {
      if (lost) {
        globalModel->overridePose(recoveryPose);
        lastFrameRecovery = true;
      } else {
        for (size_t i = 0; i < constraints.size(); i++)
          globalDeformation.addConstraint(constraints.at(i).sourcePoint, constraints.at(i).targetPoint, tick,
                                          ferns.frames.at(ferns.lastClosest)->srcTime, true);

        for (size_t i = 0; i < relativeCons.size(); i++) globalDeformation.addConstraint(relativeCons.at(i));

        assert(0);  // FIXME, input pose-graph again.
        if (globalDeformation.constrain(ferns.frames, rawGraph, tick, true, /*poseGraph,*/ true)) {
          globalModel->overridePose(recoveryPose);
          poseMatches.push_back(PoseMatch(ferns.lastClosest, ferns.frames.size(), ferns.frames.at(ferns.lastClosest)->pose,
                                          globalModel->getPose(), constraints, true));
          fernDeforms += rawGraph.size() > 0;
          fernAccepted = true;
        }
      }
    }

    // If we didn't match to a fern
    if (!lost && closeLoops && rawGraph.size() == 0) {
      // Only predict old view, since we just predicted the current view for the ferns (which failed!)
      TICK("IndexMap::INACTIVE");
      globalModel->combinedPredict(maxDepthProcessed, 0, tick - timeDelta, timeDelta, ModelProjection::INACTIVE);
      TOCK("IndexMap::INACTIVE");

      // WARNING initICP* must be called before initRGB*
      // RGBDOdometry& modelToModel = globalModel->getModelToModelOdometry();
      ModelProjection& indexMap = globalModel->getIndexMap();
      modelToModel.initICPModel(indexMap.getOldVertexTex(), indexMap.getOldNormalTex(), maxDepthProcessed, globalModel->getPose());
      modelToModel.initRGBModel(indexMap.getOldImageTex());

      modelToModel.initICP(indexMap.getSplatVertexConfTex(), indexMap.getSplatNormalTex(), maxDepthProcessed);
      modelToModel.initRGB(indexMap.getSplatImageTex());

      Eigen::Vector3f trans = globalModel->getPose().topRightCorner(3, 1);
      Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rot = globalModel->getPose().topLeftCorner(3, 3);

      modelToModel.getIncrementalTransformation(trans, rot, false, 10, pyramid, fastOdom, false, 0, 0);

      Eigen::MatrixXd covar = modelToModel.getCovariance();
      bool covOk = true;

      for (int i = 0; i < 6; i++) {
        if (covar(i, i) > covThresh) {
          covOk = false;
          break;
        }
      }

      Eigen::Matrix4f estPose = Eigen::Matrix4f::Identity();

      estPose.topRightCorner(3, 1) = trans;
      estPose.topLeftCorner(3, 3) = rot;

      if (covOk && modelToModel.lastICPCount > icpCountThresh && modelToModel.lastICPError < icpErrThresh) {
        resize.vertex(indexMap.getSplatVertexConfTex(), consBuff);
        resize.time(indexMap.getOldTimeTex(), timesBuff);

        for (int i = 0; i < consBuff.cols; i++) {
          for (int j = 0; j < consBuff.rows; j++) {
            if (consBuff.at<Eigen::Vector4f>(j, i)(2) > 0 && consBuff.at<Eigen::Vector4f>(j, i)(2) < maxDepthProcessed &&
                timesBuff.at<unsigned short>(j, i) > 0) {
              Eigen::Vector4f worldRawPoint =
                  globalModel->getPose() * Eigen::Vector4f(consBuff.at<Eigen::Vector4f>(j, i)(0), consBuff.at<Eigen::Vector4f>(j, i)(1),
                                                           consBuff.at<Eigen::Vector4f>(j, i)(2), 1.0f);

              Eigen::Vector4f worldModelPoint =
                  globalModel->getPose() * Eigen::Vector4f(consBuff.at<Eigen::Vector4f>(j, i)(0), consBuff.at<Eigen::Vector4f>(j, i)(1),
                                                           consBuff.at<Eigen::Vector4f>(j, i)(2), 1.0f);

              constraints.push_back(Ferns::SurfaceConstraint(worldRawPoint, worldModelPoint));

              localDeformation.addConstraint(worldRawPoint, worldModelPoint, tick, timesBuff.at<unsigned short>(j, i), deforms == 0);
            }
          }
        }

        std::vector<Deformation::Constraint> newRelativeCons;

        assert(0);
        if (localDeformation.constrain(ferns.frames, rawGraph, tick, false, /*poseGraph,*/ false, &newRelativeCons)) {
          poseMatches.push_back(
              PoseMatch(ferns.frames.size() - 1, ferns.frames.size(), estPose, globalModel->getPose(), constraints, false));

          deforms += rawGraph.size() > 0;

          globalModel->overridePose(estPose);

          for (size_t i = 0; i < newRelativeCons.size(); i += newRelativeCons.size() / 3) {
            relativeCons.push_back(newRelativeCons.at(i));
          }
        }
      }
    }

    if (!rgbOnly && trackingOk && !lost) {
      TICK("indexMap");
      for (auto model : models) model->predictIndices(tick, maxDepthProcessed, timeDelta);
      TOCK("indexMap");

      for (auto model : models) {
        model->fuse(tick, textures[GPUTexture::RGB], textures[GPUTexture::MASK], textures[GPUTexture::DEPTH_METRIC],
                    textures[GPUTexture::DEPTH_METRIC_FILTERED], maxDepthProcessed, weightMultiplier);
      }

      TICK("indexMap");
      for (auto model : models) model->predictIndices(tick, maxDepthProcessed, timeDelta);
      TOCK("indexMap");

      // If we're deforming we need to predict the depth again to figure out which
      // points to update the timestamp's of, since a deformation means a second pose update
      // this loop
      if (rawGraph.size() > 0 && !fernAccepted) {
        globalModel->getIndexMap().synthesizeDepth(globalModel->getPose(), globalModel->getModel(), maxDepthProcessed, initConfThresGlobal,
                                                   tick, tick - timeDelta, std::numeric_limits<unsigned short>::max());
      }

      for (auto model : models) {
        model->clean(tick, rawGraph, timeDelta, maxDepthProcessed, fernAccepted, textures[GPUTexture::DEPTH_METRIC_FILTERED],
                     textures[GPUTexture::MASK]);
      }
    }
  }

  // Update index-map textures
  predict();

  if (!lost) {
    // processFerns(); FIXME
    tick++;
  }

  moveNewModelToList();

  bool first = true;

  for (auto model : models) {
    if (model->isLoggingPoses()) {
      auto pose = first ? globalModel->getPose() :                          // cam->world
                      globalModel->getPose() * model->getPose().inverse();  // obj->world

      Eigen::Vector3f transObject = pose.topRightCorner(3, 1);
      Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rotObject = pose.topLeftCorner(3, 3);
      Eigen::Quaternionf q(rotObject);

      Eigen::Matrix<float, 7, 1> p;
      p << transObject(0), transObject(1), transObject(2), q.x(), q.y(), q.z(), q.w();
      // model->getPoseLog().push_back({tick-1, p}); //Log ticks
      model->getPoseLog().push_back({frame.timestamp, p});  // Log timestamps
    }
    first = false;
  }

  TOCK("Run");

  return false;
}

void CoFusion::processFerns() {
  TICK("Ferns::addFrame");
  ferns.addFrame(globalModel->getFillInImageTexture(), globalModel->getFillInVertexTexture(), globalModel->getFillInNormalTexture(),
                 globalModel->getPose(), tick, fernThresh);
  TOCK("Ferns::addFrame");
}

void CoFusion::predict() {
  TICK("IndexMap::ACTIVE");

  for (auto& model : models) {
    // Predict textures based on the current pose estimate
    model->combinedPredict(maxDepthProcessed, lastFrameRecovery ? 0 : tick, tick, timeDelta, ModelProjection::ACTIVE);

    // Generate textures that fill holes in predicted data with raw data (if enabled by model, currently only global model)
    model->performFillIn(textures[GPUTexture::RGB], textures[GPUTexture::DEPTH_METRIC_FILTERED], frameToFrameRGB, lost);
  }

  TOCK("IndexMap::ACTIVE");
}

bool CoFusion::requiresFillIn(ModelPointer model, float ratio) {
  if (!model->allowsFillIn()) return false;

  TICK("autoFill");
  resize.image(model->getRGBProjection(), imageBuff);
  int sum = 0;

  // TODO do this faster
  for (int i = 0; i < imageBuff.rows; i++) {
    for (int j = 0; j < imageBuff.cols; j++) {
      sum += imageBuff.at<Eigen::Matrix<unsigned char, 3, 1>>(i, j)(0) > 0 &&
             imageBuff.at<Eigen::Matrix<unsigned char, 3, 1>>(i, j)(1) > 0 && imageBuff.at<Eigen::Matrix<unsigned char, 3, 1>>(i, j)(2) > 0;
    }
  }
  TOCK("autoFill");

  // Checks whether less than ratio (75%) of the pixels are set
  return float(sum) / float(imageBuff.rows * imageBuff.cols) < ratio;
}

void CoFusion::filterDepth() {
  std::vector<Uniform> uniforms;
  uniforms.push_back(Uniform("cols", (float)Resolution::getInstance().cols()));
  uniforms.push_back(Uniform("rows", (float)Resolution::getInstance().rows()));
  uniforms.push_back(Uniform("maxD", depthCutoff));
  computePacks[ComputePack::FILTER]->compute(textures[GPUTexture::DEPTH_METRIC]->texture,
                                             &uniforms);  // Writes to GPUTexture::DEPTH_METRIC_FILTERED
}

void CoFusion::normaliseDepth(const float& minVal, const float& maxVal) {
  std::vector<Uniform> uniforms;
  uniforms.push_back(Uniform("maxVal", maxVal));
  uniforms.push_back(Uniform("minVal", minVal));
  computePacks[ComputePack::NORM_DEPTH]->compute(textures[GPUTexture::DEPTH_METRIC]->texture,
                                                 &uniforms);  // Writes to GPUTexture::DEPTH_NORM
}

void CoFusion::coloriseMasks() {
  computePacks[ComputePack::COLORISE_MASKS]->compute(textures[GPUTexture::MASK]->texture);  // Writes to GPUTexture::MASK_COLOR
}

void CoFusion::spawnObjectModel() {
  assert(!newModel);
  if (preallocatedModels.size()) {
    newModel = preallocatedModels.front();
    preallocatedModels.pop_front();
  } else {
    newModel = std::make_shared<Model>(getNextModelID(true), initConfThresObject, false, true, enablePoseLogging, modelMatchingType);
  }
  newModel->getFrameOdometry().initFirstRGB(textures[GPUTexture::RGB]);
}

bool CoFusion::redetectModels(const FrameData& frame, const SegmentationResult& segmentationResult) {
  // [Removed code]
  return false;
}

void CoFusion::moveNewModelToList() {
  if (newModel) {
    models.push_back(newModel);
    newModelListeners.callListenersDirect(newModel);
    newModel.reset();
  }
}

ModelListIterator CoFusion::inactivateModel(const ModelListIterator& it) {
  std::shared_ptr<Model> m = *it;
  std::cout << "Deactivating model... ";
  if (!enableSmartModelDelete || (m->lastCount() >= modelKeepMinSurfels && m->getConfidenceThreshold() > modelKeepConfThreshold)) {
    std::cout << "keeping data";
    // [Removed code]
    inactiveModels.push_back(m);
  } else {
    std::cout << "deleting data";
  }
  std::cout << ". Surfels: " << m->lastCount() << " confidence threshold: " << m->getConfidenceThreshold() << std::endl;

  inactiveModelListeners.callListenersDirect(m);
  return --models.erase(it);
}

unsigned char CoFusion::getNextModelID(bool assign) {
  unsigned char next = nextID;
  if (assign) {
    if (models.size() == 256)
      throw std::range_error(
          "getNextModelID(): Maximum amount of models is "
          "already in use (256).");
    while (true) {
      nextID++;
      bool isOccupied = false;
      for (auto& m : models)
        if (nextID == m->getID()) isOccupied = true;
      if (!isOccupied) break;
    }
  }
  return next;
}

void CoFusion::savePly() {
  std::cout << "Exporting PLYs..." << std::endl;

  auto exportModelPLY = [this](ModelPointer& model) {

    std::string filename = exportDir + "cloud-" + std::to_string(model->getID()) + ".ply";
    std::cout << "Storing PLY-cloud to " << filename << std::endl;

    // Open file
    std::ofstream fs;
    fs.open(filename.c_str());

    Model::SurfelMap surfelMap = model->downloadMap();
    surfelMap.countValid(model->getConfidenceThreshold());

    std::cout << "Extarcted " << surfelMap.numValid << " out of " << surfelMap.numPoints << " points." << std::endl;

    // Write header
    fs << "ply";
    fs << "\nformat "
       << "binary_little_endian"
       << " 1.0";

    // Vertices
    fs << "\nelement vertex " << surfelMap.numValid;
    fs << "\nproperty float x"
          "\nproperty float y"
          "\nproperty float z";

    fs << "\nproperty uchar red"
          "\nproperty uchar green"
          "\nproperty uchar blue";

    fs << "\nproperty float nx"
          "\nproperty float ny"
          "\nproperty float nz";

    fs << "\nproperty float radius";

    fs << "\nend_header\n";

    // Close the file
    fs.close();

    // Open file in binary appendable
    std::ofstream fpout(filename.c_str(), std::ios::app | std::ios::binary);

    Eigen::Vector4f center(0, 0, 0, 0);
    Eigen::Matrix4f gP = globalModel->getPose();
    Eigen::Matrix4f Tp = gP * model->getPose().inverse();
    Eigen::Matrix4f Tn = Tn.inverse().transpose();

    for (unsigned int i = 0; i < surfelMap.numPoints; i++) {
      Eigen::Vector4f pos = (*surfelMap.data)[(i * 3) + 0];
      float conf = pos[3];
      pos[3] = 1;

      if (conf > model->getConfidenceThreshold()) {
        Eigen::Vector4f col = (*surfelMap.data)[(i * 3) + 1];
        Eigen::Vector4f nor = (*surfelMap.data)[(i * 3) + 2];
        center += pos;
        pos = Tp * pos;
        float radius = nor[3];
        nor[3] = 0;
        nor = Tn * nor;

        nor[0] *= -1;
        nor[1] *= -1;
        nor[2] *= -1;

        float value;
        memcpy(&value, &pos[0], sizeof(float));
        fpout.write(reinterpret_cast<const char*>(&value), sizeof(float));

        memcpy(&value, &pos[1], sizeof(float));
        fpout.write(reinterpret_cast<const char*>(&value), sizeof(float));

        memcpy(&value, &pos[2], sizeof(float));
        fpout.write(reinterpret_cast<const char*>(&value), sizeof(float));

        unsigned char r = int(col[0]) >> 16 & 0xFF;
        unsigned char g = int(col[0]) >> 8 & 0xFF;
        unsigned char b = int(col[0]) & 0xFF;

        fpout.write(reinterpret_cast<const char*>(&r), sizeof(unsigned char));
        fpout.write(reinterpret_cast<const char*>(&g), sizeof(unsigned char));
        fpout.write(reinterpret_cast<const char*>(&b), sizeof(unsigned char));

        memcpy(&value, &nor[0], sizeof(float));
        fpout.write(reinterpret_cast<const char*>(&value), sizeof(float));

        memcpy(&value, &nor[1], sizeof(float));
        fpout.write(reinterpret_cast<const char*>(&value), sizeof(float));

        memcpy(&value, &nor[2], sizeof(float));
        fpout.write(reinterpret_cast<const char*>(&value), sizeof(float));

        memcpy(&value, &radius, sizeof(float));
        fpout.write(reinterpret_cast<const char*>(&value), sizeof(float));
      }
    }

    center /= surfelMap.numValid;
    std::cout << "Exported model with center: \n" << center << std::endl;

    // Close file
    fs.close();
  };

  for (auto& m : models) exportModelPLY(m);
}

void CoFusion::exportPoses() {
  std::cout << "Exporting poses..." << std::endl;

  auto exportModelPoses = [&](ModelList list) {
    for (auto& m : list) {
      if (!m->isLoggingPoses()) continue;
      std::string filename = exportDir + "poses-" + std::to_string(m->getID()) + ".txt";
      std::cout << "Storing poses to " << filename << std::endl;

      std::ofstream fs;
      fs.open(filename.c_str());

      auto poseLog = m->getPoseLog();
      for (auto& p : poseLog) {
        fs << p.ts;
        for (int i = 0; i < p.p.size(); ++i) fs << " " << p.p(i);
        fs << "\n";
      }

      fs.close();
    }
  };

  exportModelPoses(models);
  exportModelPoses(inactiveModels);
}

// Sad times ahead
ModelProjection& CoFusion::getIndexMap() { return globalModel->getIndexMap(); }

std::shared_ptr<Model> CoFusion::getBackgroundModel() { return globalModel; }

std::list<std::shared_ptr<Model>>& CoFusion::getModels() { return models; }

Ferns& CoFusion::getFerns() { return ferns; }

Deformation& CoFusion::getLocalDeformation() { return localDeformation; }

std::map<std::string, GPUTexture*>& CoFusion::getTextures() { return textures; }

const std::vector<PoseMatch>& CoFusion::getPoseMatches() { return poseMatches; }

const RGBDOdometry& CoFusion::getModelToModel() { return modelToModel; }

void CoFusion::setRgbOnly(const bool& val) { rgbOnly = val; }

void CoFusion::setIcpWeight(const float& val) { icpWeight = val; }

void CoFusion::setOutlierCoefficient(const float& val) { Model::GPUSetup::getInstance().outlierCoefficient = val; }

void CoFusion::setPyramid(const bool& val) { pyramid = val; }

void CoFusion::setFastOdom(const bool& val) { fastOdom = val; }

void CoFusion::setSo3(const bool& val) { so3 = val; }

void CoFusion::setFrameToFrameRGB(const bool& val) { frameToFrameRGB = val; }

void CoFusion::setModelSpawnOffset(const unsigned& val) { modelSpawnOffset = val; }

void CoFusion::setModelDeactivateCount(const unsigned& val) { modelDeactivateCount = val; }

void CoFusion::setCrfPairwiseSigmaRGB(const float& val) { labelGenerator.setPairwiseSigmaRGB(val); }

void CoFusion::setCrfPairwiseSigmaPosition(const float& val) { labelGenerator.setPairwiseSigmaPosition(val); }

void CoFusion::setCrfPairwiseSigmaDepth(const float& val) { labelGenerator.setPairwiseSigmaDepth(val); }

void CoFusion::setCrfPairwiseWeightAppearance(const float& val) { labelGenerator.setPairwiseWeightAppearance(val); }

void CoFusion::setCrfPairwiseWeightSmoothness(const float& val) { labelGenerator.setPairwiseWeightSmoothness(val); }

void CoFusion::setCrfThresholdNew(const float& val) { labelGenerator.setUnaryThresholdNew(val); }

void CoFusion::setCrfUnaryWeightError(const float& val) { labelGenerator.setUnaryWeightError(val); }

void CoFusion::setCrfIteration(const unsigned& val) { labelGenerator.setIterationsCRF(val); }

void CoFusion::setCrfUnaryKError(const float& val) { labelGenerator.setUnaryKError(val); }

void CoFusion::setNewModelMinRelativeSize(const float& val) { labelGenerator.setNewModelMinRelativeSize(val); }

void CoFusion::setNewModelMaxRelativeSize(const float& val) { labelGenerator.setNewModelMaxRelativeSize(val); }

void CoFusion::setFernThresh(const float& val) { fernThresh = val; }

void CoFusion::setDepthCutoff(const float& val) { depthCutoff = val; }

const bool& CoFusion::getLost() {  // lel
  return lost;
}

const int& CoFusion::getTick() { return tick; }

const int& CoFusion::getTimeDelta() { return timeDelta; }

void CoFusion::setTick(const int& val) { tick = val; }

const float& CoFusion::getMaxDepthProcessed() { return maxDepthProcessed; }

const Eigen::Matrix4f& CoFusion::getCurrPose() { return globalModel->getPose(); }

const int& CoFusion::getDeforms() { return deforms; }

const int& CoFusion::getFernDeforms() { return fernDeforms; }

std::map<std::string, FeedbackBuffer*>& CoFusion::getFeedbackBuffers() { return feedbackBuffers; }
