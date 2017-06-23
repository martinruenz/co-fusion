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

#pragma once

#include "Utils/Macros.h"
#include "Utils/RGBDOdometry.h"
#include "Utils/Resolution.h"
#include "Utils/Intrinsics.h"
#include "Utils/Stopwatch.h"
#include "Callbacks.h"
#include "Shaders/Shaders.h"
#include "Shaders/ComputePack.h"
#include "Shaders/FeedbackBuffer.h"
#include "Shaders/FillIn.h"
#include "Model/Deformation.h"
#include "Model/Model.h"
#include "Model/ModelProjection.h"
#include "Ferns.h"
#include "PoseMatch.h"
#include "FrameData.h"
#include "Segmentation/Segmentation.h"

#include <list>
#include <iomanip>
#include <memory>
#include <pangolin/gl/glcuda.h>

class CoFusion {
 public:
  CoFusion(const int timeDelta = 200, const int countThresh = 35000, const float errThresh = 5e-05, const float covThresh = 1e-05,
           const bool closeLoops = true, const bool iclnuim = false, const bool reloc = false, const float photoThresh = 115,
           const float initConfidenceGlobal = 4, const float initConfidenceObject = 2, const float depthCut = 3, const float icpThresh = 10,
           const bool fastOdom = false, const float fernThresh = 0.3095, const bool so3 = true, const bool frameToFrameRGB = false,
           const unsigned modelSpawnOffset = 20, const Model::MatchingType matchingType = Model::MatchingType::Drost,
           const std::string& exportDirectory = "", const bool exportSegmentationResults = false);

  virtual ~CoFusion();

  void preallocateModels(unsigned count);

  SegmentationResult performSegmentation(const FrameData& frame);

  /**
       * Process an rgb/depth map pair
       * @param frame Frame data (rgb, depth, time)
       * @param inPose optional input SE3 pose (if provided, we don't attempt to perform tracking)
       * @param weightMultiplier optional full frame fusion weight
       * @param bootstrap if true, use inPose as a pose guess rather than replacement
       * @return returns true if a pause might be interesting, can be ignored without hesitation
       */
  bool processFrame(const FrameData& frame, const Eigen::Matrix4f* inPose = 0, const float weightMultiplier = 1.f,
                    const bool bootstrap = false);

  /**
       * Predicts the current view of the scene, updates the [vertex/normal/image]Tex() members
       * of the indexMap class
       */
  void predict();

  /**
       * This class contains all of the predicted renders
       * @return reference
       */
  ModelProjection& getIndexMap();

  /**
       * This class contains the surfel map
       * @return
       */
  std::shared_ptr<Model> getBackgroundModel();

  std::list<std::shared_ptr<Model>>& getModels();

  /**
       * This class contains the fern keyframe database
       * @return
       */
  Ferns& getFerns();

  /**
       * This class contains the local deformation graph
       * @return
       */
  Deformation& getLocalDeformation();

  /**
       * This is the map of raw input textures (you can display these)
       * @return
       */
  std::map<std::string, GPUTexture*>& getTextures();

  /**
       * This is the list of deformation constraints
       * @return
       */
  const std::vector<PoseMatch>& getPoseMatches();

  /**
       * This is the tracking class, if you want access
       * @return
       */
  const RGBDOdometry& getModelToModel();

  /**
       * The point fusion confidence threshold
       * @return
       */
  const float& getConfidenceThreshold();

  /**
       * If you set this to true we just do 2.5D RGB-only Lucas–Kanade tracking (no fusion)
       * @param val
       */
  void setRgbOnly(const bool& val);

  /**
       * Weight for ICP in tracking
       * @param val if 100, only use depth for tracking, if 0, only use RGB. Best value is 10
       */
  void setIcpWeight(const float& val);

  void setOutlierCoefficient(const float& val);

  /**
       * Whether or not to use a pyramid for tracking
       * @param val default is true
       */
  void setPyramid(const bool& val);

  /**
       * Controls the number of tracking iterations
       * @param val default is false
       */
  void setFastOdom(const bool& val);

  /**
       * Turns on or off SO(3) alignment bootstrapping
       * @param val
       */
  void setSo3(const bool& val);

  /**
       * Turns on or off frame to frame tracking for RGB
       * @param val
       */
  void setFrameToFrameRGB(const bool& val);

  /**
       * Raw data fusion confidence threshold
       * @param val default value is 10, but you can play around with this
       */
  void setConfidenceThreshold(const float& val);

  /**
       * Threshold for sampling new keyframes
       * @param val default is some magic value, change at your own risk
       */
  void setFernThresh(const float& val);

  /**
       * Cut raw depth input off at this point
       * @param val default is 3 meters
       */
  void setDepthCutoff(const float& val);

  /**
       * Returns whether or not the camera is lost, if relocalisation mode is on
       * @return
       */
  const bool& getLost();

  /**
       * Get the internal clock value of the fusion process
       * @return monotonically increasing integer value (not real-world time)
       */
  const int& getTick();

  /**
       * Get the time window length for model matching
       * @return
       */
  const int& getTimeDelta();

  /**
       * Cheat the clock, only useful for multisession/log fast forwarding
       * @param val control time itself!
       */
  void setTick(const int& val);

  /**
       * Internal maximum depth processed, this is defaulted to 20 (for rescaling depth buffers)
       * @return
       */
  const float& getMaxDepthProcessed();

  /**
       * The current global camera pose estimate
       * @return SE3 pose
       */
  const Eigen::Matrix4f& getCurrPose();

  /**
       * The number of local deformations that have occurred
       * @return
       */
  const int& getDeforms();

  /**
       * The number of global deformations that have occured
       * @return
       */
  const int& getFernDeforms();

  // void setExportSegmentationDirectory(const std::string& path) { exportSegDir = path; }

  void setModelSpawnOffset(const unsigned& val);
  void setModelDeactivateCount(const unsigned& val);
  void setCrfPairwiseSigmaRGB(const float& val);
  void setCrfPairwiseSigmaPosition(const float& val);
  void setCrfPairwiseSigmaDepth(const float& val);
  void setCrfPairwiseWeightAppearance(const float& val);
  void setCrfPairwiseWeightSmoothness(const float& val);
  void setCrfThresholdNew(const float& val);
  void setCrfUnaryWeightError(const float& val);
  void setCrfIteration(const unsigned& val);
  void setCrfUnaryKError(const float& val);
  void setNewModelMinRelativeSize(const float& val);
  void setNewModelMaxRelativeSize(const float& val);
  void setEnableMultipleModels(bool val) { enableMultipleModels = val; }
  void setEnableSmartModelDelete(bool val) { enableSmartModelDelete = val; }
  // void setCrfUnaryWeightErrorBackground(const float& val);
  // void setCrfUnaryWeightConfBackground(const float& val);

  /**
       * These are the vertex buffers computed from the raw input data
       * Each of these buffers stores one vertex per pixel
       * @return can be rendered
       */
  std::map<std::string, FeedbackBuffer*>& getFeedbackBuffers();

  /**
       * Calculate the above for the current frame (only done on the first frame normally)
       */
  void computeFeedbackBuffers();

  /**
   * Saves out a .ply mesh file of the current model
   */
  void savePly();

  void exportPoses();

  /**
       * Renders a normalised view of the input metric depth for displaying as an OpenGL texture
       * (this is stored under textures[GPUTexture::DEPTH_NORM]
       * @param minVal minimum depth value to render
       * @param maxVal maximum depth value to render
       */
  void normaliseDepth(const float& minVal, const float& maxVal);

  /**
       * Renders a colorised view of the segmentation (masks)
       * (this is stored under textures[GPUTexture::MASK_COLOR]
       */
  void coloriseMasks();

  // Listeners

  /// Called when a new model is created
  inline void addNewModelListener(const ModelListener& listener) { newModelListeners.addListener(listener); }

  /// Called when a model becomes inactive
  inline void addInactiveModelListener(const ModelListener& listener) { inactiveModelListeners.addListener(listener); }

  // Here be dragons
 private:
  void spawnObjectModel();
  bool redetectModels(const FrameData& frame, const SegmentationResult& segmentationResult);
  void moveNewModelToList();
  ModelListIterator inactivateModel(const ModelListIterator& it);

  unsigned char getNextModelID(bool assign = false);

  void createTextures();
  void createCompute();
  void createFeedbackBuffers();

  void filterDepth();

  // Should raw data (of last frame) be added to the rendered vertexTexture / normalTexture / imageTexture (from last pose)?
  bool requiresFillIn(ModelPointer model, float ratio = 0.75f);

  void processFerns();

 private:
  ModelList models;  // also contains static environment (first model)
  ModelList inactiveModels;
  ModelList preallocatedModels;        // Since some systems are slow in allocating new models, allow to preallocate models for later use.
  ModelPointer newModel;               // Stored and added to list as soon as frame is processed
  std::shared_ptr<Model> globalModel;  // static environment
  unsigned char nextID = 0;
  Segmentation labelGenerator;

  Model::MatchingType modelMatchingType;

  CallbackBuffer<std::shared_ptr<Model>> newModelListeners;
  CallbackBuffer<std::shared_ptr<Model>> inactiveModelListeners;

  RGBDOdometry modelToModel;

  // TODO move to model?
  Ferns ferns;
  Deformation localDeformation;
  Deformation globalDeformation;

  std::map<std::string, GPUTexture*> textures;
  std::map<std::string, ComputePack*> computePacks;
  std::map<std::string, FeedbackBuffer*> feedbackBuffers;

  int tick;
  const int timeDelta;
  const int icpCountThresh;
  const float icpErrThresh;
  const float covThresh;

  int deforms;
  int fernDeforms;
  const int consSample;
  GPUResize resize;

  std::vector<PoseMatch> poseMatches;
  std::vector<Deformation::Constraint> relativeCons;

  // std::vector<std::pair<int64_t, Eigen::Matrix4f> > poseGraph;
  // std::vector<int64_t> poseLogTimes;

  // Scaled down textures
  Img<Eigen::Matrix<unsigned char, 3, 1>> imageBuff;  // imageTex (only used for fillIn)
  Img<Eigen::Vector4f> consBuff;                      // vertexTex
  Img<unsigned short> timesBuff;                      // oldTimeTex

  const bool closeLoops;
  const bool iclnuim;

  const bool reloc;
  bool lost;
  bool lastFrameRecovery;
  int trackingCount;
  const float maxDepthProcessed;

  bool enableMultipleModels = true;
  bool enableSmartModelDelete = true;
  bool enableRedetection = false;
  bool enableModelMerging = false;
  bool enableSpawnSubtraction = true;
  bool enablePoseLogging = true;
  bool rgbOnly;
  float icpWeight;
  bool pyramid;
  bool fastOdom;
  float initConfThresGlobal;
  float initConfThresObject;
  float fernThresh;
  bool so3;
  bool frameToFrameRGB;
  float depthCutoff;
  unsigned modelDeactivateCount = 10;   // deactivate model, when not seen for this many frames FIXME unused
  unsigned modelKeepMinSurfels = 4000;  // Only keep deactivated models with at least this many surfels
  float modelKeepConfThreshold = 0.3;
  unsigned modelSpawnOffset;  // setting
  unsigned spawnOffset = 0;   // current value

  bool exportSegmentation;
  std::string exportDir;
};
