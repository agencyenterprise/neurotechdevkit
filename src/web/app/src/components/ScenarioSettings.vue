<template>
  <div class="mb-3">
    <div class="btn-group">
      <input :disabled="hasSimulation || isRunningSimulation" class="btn-check" type="radio" value="preBuilt"
        id="preBuilt" v-model="scenarioType" name="scenarioType" />
      <label class="btn btn-primary btn-wide" for="preBuilt"
        v-tooltip="'Use a prebuilt scenario with a target and transducers'">Prebuilt</label>
      <input :disabled="hasSimulation || isRunningSimulation" class="btn-check" type="radio" id="ctFile" value="ctFile"
        v-model="scenarioType" name="scenarioType" />
      <label class="btn btn-primary btn-wide" for="ctFile" v-tooltip="'Load a CT scan from a file'">CT Scan</label>
    </div>
  </div>
  <div class="mb-3">
    <div class="mb-3" v-if="isPreBuilt">
      <label class="form-label">Scenarios:</label>
      <select :disabled="hasSimulation || isRunningSimulation" class="form-select" v-model="selectedScenario">
        <option disabled value="null">Select a scenario</option>
        <option v-for="(scenario, key) in currentBuiltInScenarios" :key="key" :value="key">
          {{ scenario.title }}
        </option>
      </select>
    </div>
    <div v-else>

      <div class="mb-3">
        <label class="form-label">Available CTs</label>
        <select class="form-select" size="3" v-model="ctFile">
          <option v-for="ct in availableCTs" :key="ct.filename" :value="ct.filename">{{ ct.filename }}</option>
        </select>
      </div>
      <div class="mb-3" v-if="is2d">
        <label class="form-label" v-tooltip="'Axis along which to slice the 3D field to be recorded'">Axis</label>
        <select class="form-select" aria-label="Axis" v-model="ctSliceAxis">
          <option disabled value="">Select an axis</option>
          <option value="x">X</option>
          <option value="y">Y</option>
          <option value="z">Z</option>
        </select>
      </div>
      <div class="mb-3" v-if="is2d">
        <label class="form-label"
          v-tooltip="'The position along the slice axis at which the slice of the 3D field should be made'">Distance
          from
          origin (m)</label>
        <input type="number" step="any" class="form-control" placeholder="0.0" v-model.number="ctSlicePosition" />
      </div>
      <div class="mb-3">
        <label class="form-label"
          v-tooltip="'Select the CT file and the file containing the mapping between layers and masks'">CT and mapping
          files</label>
        <input class="form-control" type="file" ref="ctFilesInput" @change="filesChosen" multiple />
      </div>
      <div class="mb-3">
        <button class="btn btn-primary" type="button" @click="fileUploadClicked" :disabled="!filesReady">
          Upload new CT
        </button>
      </div>
    </div>
  </div>
</template>
<script>
import { mapGetters, mapActions } from 'vuex';

export default {
  data() {
    return {
      filesReady: false,
    };
  },
  computed: {
    ...mapGetters('scenarioSettings', [
      'availableCTs',
      'isPreBuilt',
      'builtInScenarios2d',
      'builtInScenarios3d',
      'scenarioId',
    ]),
    ...mapGetters(['is2d', 'hasSimulation', 'isRunningSimulation']),
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
    selectedScenario: {
      get() {
        const scenarios = this.is2d ? this.builtInScenarios2d : this.builtInScenarios3d;
        return this.scenarioId in scenarios ? this.scenarioId : null;
      },
      set(value) {
        this.setBuiltinScenario(value);
      }
    },

    ctFile: {
      get() {
        return this.$store.getters['scenarioSettings/ctFile'];
      },
      set(value) {
        this.setCTFile(value);
      }
    },
    ctSliceAxis: {
      get() {
        return this.$store.getters['scenarioSettings/ctSliceAxis'];
      },
      set(value) {
        this.setCtSliceAxis(value);
      }
    },
    ctSlicePosition: {
      get() {
        return this.$store.getters['scenarioSettings/ctSlicePosition'];
      },
      set(value) {
        this.setCtSlicePosition(value);
      }
    }
  },
  methods: {
    ...mapActions(['setIsProcessing', 'setProcessingMessage']),
    ...mapActions('scenarioSettings', [
      'setBuiltinScenario',
      'setCTFile',
      'setAvailableCTs',
      'setIsPreBuilt',
      'setCtSliceAxis',
      'setCtSlicePosition'
    ]),
    filesChosen(event) {
      this.filesReady = event.target.files.length >= 2; // Ensure there are at least two files selected
    },
    fileUploadClicked() {
      const input = this.$refs.ctFilesInput; // Access the file input via ref

      if (input.files.length >= 2) {
        const data = new FormData();
        data.append('file_0', input.files[0], input.files[0].name);
        data.append('file_1', input.files[1], input.files[1].name);

        this.setIsProcessing(true);
        this.setProcessingMessage('Uploading CT scan...');
        fetch(`http://${process.env.VUE_APP_BACKEND_URL}/ct_scan`, {
          method: 'POST',
          body: data
        })
          .then(response => {
            if (!response.ok) {
              throw new Error('Network response was not ok');
            }
            return response.json();
          })
          .then(message => {
            console.log('Success:', message);
            this.setAvailableCTs(message.available_cts)
            this.setCTFile(message.selected_ct.filename);
            this.filesReady = false; // Reset files readiness
            this.setIsProcessing(false);
          })
          .catch(error => {
            console.error('Error:', error);
            this.filesReady = false; // Reset files readiness
            this.setIsProcessing(false);
          });
      }
    },
  }
};
</script>
<style scoped></style>