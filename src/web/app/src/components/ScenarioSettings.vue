<template>
  <div class="mb-3">
    <div class="btn-group">
      <input :disabled="hasSimulation" class="btn-check" type="radio" value="preBuilt" id="preBuilt"
        v-model="scenarioType" />
      <label class="btn btn-primary btn-wide" for="preBuilt"
        title="Use a prebuilt scenario with a target and transducers">Prebuilt</label>
      <input :disabled="hasSimulation" class="btn-check" type="radio" id="ctScan" value="ctScan"
        v-model="scenarioType" />
      <label class="btn btn-primary btn-wide" for="ctScan" title="Load a CT scan from a file">CT Scan</label>
    </div>
  </div>
  <div class="mb-3">
    <div class="mb-3" v-if="isPreBuilt">
      <label class="form-label">Scenarios:</label>
      <select v-if="is2d" :disabled="hasSimulation" class="form-select" v-model="selected2dScenario">
        <option disabled value="null">Select a scenario</option>
        <option v-for="(scenario, key) in builtInScenarios2d" :key="key" :value="key">
          {{ scenario.title }}
        </option>
      </select>
      <select v-else :disabled="hasSimulation" class="form-select" v-model="selected3dScenario">
        <option disabled value="null">Select a scenario</option>
        <option v-for="(scenario, key) in builtInScenarios3d" :key="key" :value="key">
          {{ scenario.title }}
        </option>
      </select>
    </div>
    <div v-else>
      <form @submit.prevent="uploadCTScan">
        <div class="mb-3">
          <label class="form-label">Available CTs</label>
          <select class="form-select" size="3" v-model="selectedCTScan">
            <option v-for="ct in ctScans" :key="ct.filename" :value="ct.filename">{{ ct.filename }}</option>
          </select>
        </div>
        <div class="mb-3" v-if="is2d">
          <label class="form-label" title="Axis along which to slice the 3D field to be recorded">Axis</label>
          <select class="form-select" aria-label="Axis" v-model="ctSliceAxis">
            <option disabled value="">Select an axis</option>
            <option value="x">X</option>
            <option value="y">Y</option>
            <option value="z">Z</option>
          </select>
        </div>
        <div class="mb-3" v-if="is2d">
          <label class="form-label"
            title="The position along the slice axis at which the slice of the 3D field should be made">Distance from
            origin (m)</label>
          <input type="number" step="any" class="form-control" placeholder="0.0" v-model.number="ctSlicePosition" />
        </div>
      </form>
      <div class="mb-3">
        <label class="form-label"
          title="Select the CT file and the file containing the mapping between layers and masks">CT and mapping
          Files</label>
        <input class="form-control" type="file" @change="filesChosen" multiple />
      </div>
      <div class="mb-3">
        <button class="btn btn-primary" type="button" @click="fileUploadClicked" :disabled="!filesReady">Upload new
          CT</button>
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
      ctSliceAxis: '',
      ctSlicePosition: 0.0,
      filesReady: false,
    };
  },
  computed: {
    ...mapGetters('scenarioSettings', ['ctScans', 'isPreBuilt', 'builtInScenarios2d', 'builtInScenarios3d', 'scenarioId']),
    ...mapGetters(['is2d', 'hasSimulation']),
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
    },
    selected2dScenario: {
      get() {
        // if scenarioId in builtInScenarios2d, return scenarioId
        // else return null
        if (this.scenarioId in this.builtInScenarios2d) {
          return this.scenarioId;
        }
        return null
      },
      set(value) {
        this.setScenario(value);
      }
    },
    selected3dScenario: {
      get() {
        // if scenarioId in builtInScenarios3d, return scenarioId
        // else return null
        if (this.scenarioId in this.builtInScenarios3d) {
          return this.scenarioId;
        }
        return null
      },
      set(value) {
        this.setScenario(value);
      }
    },
  },
  methods: {
    ...mapActions('scenarioSettings', ['setScenario', 'setCTScan', 'setIsPreBuilt']),
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
    selectedCTScan(newVal) {
      if (newVal) {
        this.setCTScan(newVal);
      }
    }
  }
};
</script>
<style scoped></style>