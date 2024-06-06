<template>
  <div>
    <div class="mb-3">
      <div class="btn-group">
        <input class="btn-check" type="radio" value="preBuilt" id="isPreBuilt" v-model="scenarioType" />
        <label class="btn btn-primary btn-wide" for="isPreBuilt"
          title="Use a prebuilt scenario with a target and transducers">Prebuilt</label>
        <input class="btn-check" type="radio" value="ctScan" id="isCtScan" v-model="scenarioType" />
        <label class="btn btn-primary btn-wide" for="isCtScan" title="Load a CT scan from a file">CT Scan</label>
      </div>
    </div>
    <div class="mb-3">
      <div class="mb-3" v-if="isPreBuilt">
        <label for="scenario" class="form-label">Scenarios:</label>
        <select id="scenario" class="form-select" v-model="selectedScenario">
          <option disabled value="">Select a scenario</option>
          <option v-for="(scenario, key) in currentBuiltInScenarios" :key="key" :value="key">
            {{ scenario.title }}
          </option>
        </select>
      </div>
      <div v-else>
        <form @submit.prevent="uploadCTScan">
          <div class="mb-3">
            <label class="form-label" for="loadedCTs">Available CTs</label>
            <select class="form-select" size="3" id="loadedCTs" v-model="selectedCTScan">
              <option v-for="ct in ctScans" :key="ct.filename" :value="ct.filename">{{ ct.filename }}</option>
            </select>
          </div>
          <div class="mb-3" v-if="is2d">
            <label for="ctSliceAxis" class="form-label"
              title="Axis along which to slice the 3D field to be recorded">Axis</label>
            <select class="form-select" aria-label="Axis" id="ctSliceAxis" v-model="ctSliceAxis">
              <option disabled value="">Select an axis</option>
              <option value="x">X</option>
              <option value="y">Y</option>
              <option value="z">Z</option>
            </select>
          </div>
          <div class="mb-3" v-if="is2d">
            <label for="ctSlicePosition" class="form-label"
              title="The position along the slice axis at which the slice of the 3D field should be made">Distance from
              Origin (m)</label>
            <input type="number" step="any" class="form-control" id="ctSlicePosition" placeholder="0.0"
              v-model.number="ctSlicePosition" />
          </div>
        </form>
        <div class="mb-3">
          <label for="ctFiles" class="form-label"
            title="Select the CT file and the file containing the mapping between layers and masks">CT and mapping
            Files</label>
          <input class="form-control" type="file" id="ctFiles" @change="filesChosen" multiple />
        </div>
        <div class="mb-3">
          <button id="fileUploadButton" class="btn btn-primary" type="button" @click="fileUploadClicked"
            :disabled="!filesReady">Upload new CT</button>
        </div>
      </div>
    </div>
  </div>
</template>
<script>
import { mapGetters, mapActions } from 'vuex';

export default {
  data() {
    return {
      selectedCTScan: '',
      selectedScenario: '',
      ctSliceAxis: '',
      ctSlicePosition: 0.0,
      filesReady: false,
    };
  },
  computed: {
    ...mapGetters({
      ctScans: 'ctScans',
      isPreBuilt: 'isPreBuilt',
      is2d: 'is2d',
      builtInScenarios2d: 'builtInScenarios2d',
      builtInScenarios3d: 'builtInScenarios3d'
    }),
    currentBuiltInScenarios() {
      return this.is2d ? this.builtInScenarios2d : this.builtInScenarios3d;
    },
    scenarioType: {
      get() {
        return this.isPreBuilt ? 'preBuilt' : 'ctScan';
      },
      set(value) {
        this.setIsPreBuilt(value === 'preBuilt');
      }
    }
  },
  methods: {
    ...mapActions({
      setScenario: 'setScenario',
      setCTScan: 'setCTScan',
      setIsPreBuilt: 'setIsPreBuilt'
    }),
    filesChosen(event) {
      // Handle file chosen logic
      // Update `filesReady` based on the selection
      this.filesReady = !!event.target.files.length;
    },
    fileUploadClicked() {
      // Handle file upload logic
    },
    uploadCTScan() {
      // Handle CT scan upload logic
    }
  },
  watch: {
    selectedScenario(newVal) {
      if (newVal) {
        this.setScenario(newVal);
      }
    },
    selectedCTScan(newVal) {
      if (newVal) {
        this.setCTScan(newVal);
      }
    }
  }
};
</script>
<style scoped></style>