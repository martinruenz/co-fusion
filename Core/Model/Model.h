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

#include "../Utils/RGBDOdometry.h"
#include "../Shaders/Shaders.h"
#include "../Shaders/Uniform.h"
#include "../Shaders/FillIn.h"
#include "../Shaders/FeedbackBuffer.h"
#include "../GPUTexture.h"
#include "../Utils/Resolution.h"
#include "../Utils/Stopwatch.h"
#include "../Utils/Intrinsics.h"
#include "../FrameData.h"
#include "ModelProjection.h"
#include <pangolin/gl/gl.h>
#include <memory>
#include <list>
#include <opencv2/imgproc/imgproc.hpp>

#include "Buffers.h"

class IModelMatcher;
class Model;
typedef std::shared_ptr<Model> ModelPointer;
typedef std::list<ModelPointer> ModelList;
typedef ModelList::iterator ModelListIterator;

struct ModelDetectionResult {
  // float prob.
  Eigen::Matrix4f pose;
  bool isGood;
};

class Model {
 public:
  // Shared data for each model
  struct GPUSetup {
    static GPUSetup& getInstance() {
      static GPUSetup instance;
      return instance;
    }

    // TODO: A lot of the attributes here should be either static or encapsulated elsewhere!
    std::shared_ptr<Shader> initProgram;
    std::shared_ptr<Shader> drawProgram;
    std::shared_ptr<Shader> drawSurfelProgram;

    // For supersample fusing
    std::shared_ptr<Shader> dataProgram;
    std::shared_ptr<Shader> updateProgram;
    std::shared_ptr<Shader> unstableProgram;
    std::shared_ptr<Shader> eraseProgram;
    pangolin::GlRenderBuffer renderBuffer;

    // NOTICE: The dimension of these textures suffice the VBO, not the sensor! See TEXTURE_DIMENSION
    pangolin::GlFramebuffer frameBuffer;  // Frame buffer, holding the following textures:
    GPUTexture updateMapVertsConfs;       // We render updated vertices vec3 + confidences to one texture
    GPUTexture updateMapColorsTime;       // We render updated colors vec3 + timestamps to another
    GPUTexture updateMapNormsRadii;       // We render updated normals vec3 + radii to another

    // Current depth / mask pyramid used for Odometry
    std::vector<DeviceArray2D<float>> depth_tmp;
    std::vector<DeviceArray2D<unsigned char>> mask_tmp;

    float outlierCoefficient = 0.9;

   private:
    GPUSetup();
  };

  enum class MatchingType { Drost };

 public:
  static const int TEXTURE_DIMENSION;
  static const int MAX_VERTICES;
  static const int NODE_TEXTURE_DIMENSION;
  static const int MAX_NODES;

 private:
  // static std::list<unsigned char> availableIDs;

 public:
  Model(unsigned char id, float confidenceThresh, bool enableFillIn = true, bool enableErrorRecording = true,
        bool enablePoseLogging = false, MatchingType matchingType = MatchingType::Drost,
        float maxDepth = std::numeric_limits<float>::max());  // TODO: Default disable
  virtual ~Model();

  // ----- Functions ----- //

  virtual unsigned int lastCount();
  static Eigen::Vector3f rodrigues2(const Eigen::Matrix3f& matrix);  // TODO mode to a nicer place

  // ----- Re-detection

  virtual void buildDescription();
  virtual ModelDetectionResult detectInRegion(const FrameData& frame, const cv::Rect& rect);

  // ----- Init

  virtual void initialise(const FeedbackBuffer& rawFeedback, const FeedbackBuffer& filteredFeedback);

  virtual void renderPointCloud(const Eigen::Matrix4f& vp, const bool drawUnstable, const bool drawPoints, const bool drawWindow,
                                const int colorType, const int time, const int timeDelta);

  static void generateCUDATextures(GPUTexture* depth, GPUTexture* mask);

  virtual void initICP(bool doFillIn, bool frameToFrameRGB, float depthCutoff, GPUTexture* rgb);

  // ----- Tracking and fusion

  virtual void performTracking(bool frameToFrameRGB, bool rgbOnly, float icpWeight, bool pyramid, bool fastOdom, bool so3,
                               float maxDepthProcessed, GPUTexture* rgb, int64_t logTimestamp, bool tryFillIn = false);

  // Compute fusion-weight based on velocity
  virtual float computeFusionWeight(float weightMultiplier) const;

  // Assuming the indexMap is already computed, perform fusion. 1) associate data, 2) update model
  virtual void fuse(const int& time, GPUTexture* rgb, GPUTexture* mask, GPUTexture* depthRaw, GPUTexture* depthFiltered,
                    const float depthCutoff, const float weightMultiplier);

  // Always called after fuse. Copy unstable points to map.
  virtual void clean(const int& time, std::vector<float>& graph, const int timeDelta, const float depthCutoff, const bool isFern,
                     GPUTexture* depthFiltered, GPUTexture* mask);

  // ...
  virtual void eraseErrorGeometry(GPUTexture* depthFiltered);

  // ----- Prediction and fillin

  inline bool allowsFillIn() const { return fillIn ? true : false; }

  void performFillIn(GPUTexture* rawRGB, GPUTexture* rawDepth, bool frameToFrameRGB, bool lost);

  inline void combinedPredict(float depthCutoff, int time, int maxTime, int timeDelta, ModelProjection::Prediction predictionType) {
    indexMap.combinedPredict(getPose(), getModel(), depthCutoff, getConfidenceThreshold(), time, maxTime, timeDelta, predictionType);
  }

  inline void predictIndices(int time, float depthCutoff, int timeDelta) {
    indexMap.predictIndices(getPose(), time, getModel(), depthCutoff, timeDelta);
  }

  // ----- Getter ----- //

  inline float getConfidenceThreshold() const { return confidenceThreshold; }

  inline void setConfidenceThreshold(float confThresh) { confidenceThreshold = confThresh; }

  inline void setMaxDepth(float d) { maxDepth = d; }

  // Returns a vector of 4-float tuples: position0, color0, normal0, ..., positionN, colorN, normalN
  // Where position is (x,y,z,conf), color is (color encoded as a 24-bit integer, <unused>, initTime, timestamp), and normal
  // is (x,y,z,radius)
  struct SurfelMap {
    std::unique_ptr<std::vector<Eigen::Vector4f>> data;
    void countValid(const float& confThres) {
      numValid = 0;
      for (unsigned int i = 0; i < numPoints; i++) {
        const Eigen::Vector4f& pos = (*data)[(i * 3) + 0];
        if (pos[3] > confThres) numValid++;
      }
    }
    unsigned numPoints = 0;
    unsigned numValid = 0;
  };

  virtual SurfelMap downloadMap();

  // inline cv::Mat downloadUnaryConfTexture() {
  //    return indexMap.getUnaryConfTex()->downloadTexture();
  //}

  inline cv::Mat downloadVertexConfTexture() { return indexMap.getSplatVertexConfTex()->downloadTexture(); }

  inline cv::Mat downloadICPErrorTexture() { return icpError->downloadTexture(); }
  inline cv::Mat downloadRGBErrorTexture() { return rgbError->downloadTexture(); }

  inline GPUTexture* getICPErrorTexture() { return icpError.get(); }
  inline GPUTexture* getRGBErrorTexture() { return rgbError.get(); }

  inline GPUTexture* getRGBProjection() { return indexMap.getSplatImageTex(); }
  inline GPUTexture* getVertexConfProjection() { return indexMap.getSplatVertexConfTex(); }
  inline GPUTexture* getNormalProjection() { return indexMap.getSplatNormalTex(); }
  inline GPUTexture* getTimeProjection() { return indexMap.getSplatTimeTex(); }
  // inline GPUTexture* getUnaryConfTexture() {
  //    return indexMap.getUnaryConfTex();
  //}
  inline GPUTexture* getFillInImageTexture() { return &(fillIn->imageTexture); }
  inline GPUTexture* getFillInNormalTexture() { return &(fillIn->normalTexture); }
  inline GPUTexture* getFillInVertexTexture() { return &(fillIn->vertexTexture); }

  virtual const OutputBuffer& getModel();

  inline const Eigen::Matrix4f& getPose() const { return pose; }
  inline const Eigen::Matrix4f& getLastPose() const { return lastPose; }
  inline void overridePose(const Eigen::Matrix4f& p) {
    pose = p;
    lastPose = p;
  }
  inline Eigen::Matrix4f getLastTransform() const { return getPose().inverse() * lastPose; }

  inline unsigned int getID() const { return id; }

  inline RGBDOdometry& getFrameOdometry() { return frameToModel; }
  inline ModelProjection& getIndexMap() { return indexMap; }

  inline unsigned getUnseenCount() const { return unseenCount; }
  inline void resetUnseenCount() { unseenCount = 0; }
  inline unsigned incrementUnseenCount() {
    if (unseenCount < std::numeric_limits<unsigned>::max()) return ++unseenCount;
    return unseenCount;
  }

  struct PoseLogItem {
    int64_t ts;
    Eigen::Matrix<float, 7, 1> p;  // x,y,z, qx,qy,qz,qw
  };
  inline bool isLoggingPoses() const { return poseLog.capacity() > 0; }
  inline std::vector<PoseLogItem>& getPoseLog() { return poseLog; }

 protected:
  // Current pose
  Eigen::Matrix4f pose;
  Eigen::Matrix4f lastPose;

  std::vector<PoseLogItem> poseLog;  // optional, for testing

  // Confidence Threshold (low in the beginning, increasing)
  float confidenceThreshold;
  float maxDepth;

  // surfel buffers (swapping in order to update)
  OutputBuffer vbos[2];  // Todo make dynamic buffer

  int target, renderSource;  // swapped after FUSE and CLEAN

  // MAX_VERTICES * Vertex::SIZE, currently 3072*3072 vertices
  static const int bufferSize;

  // Count surfels in model
  GLuint countQuery;
  unsigned int count;

  // 16 floats stored column-major yo'
  static GPUTexture deformationNodes;  // Todo outsource to derived class?

  OutputBuffer newUnstableBuffer;
  // Vbo, newUnstableFid;

  GLuint uvo;  // One element for each pixel (size=width*height), used as "layout (location = 0) in vec2 texcoord;" in data.vert
  int uvSize;
  unsigned int id;

  std::unique_ptr<GPUTexture> icpError;
  std::unique_ptr<GPUTexture> rgbError;

  const GPUSetup& gpu;

  ModelProjection indexMap;
  RGBDOdometry frameToModel;

  unsigned unseenCount = 0;

  // Fill in holes in prediction (using raw data)
  std::unique_ptr<FillIn> fillIn;

  // Allows to detect inactive models in depth-map-region
  std::unique_ptr<IModelMatcher> modelMatcher;
};
