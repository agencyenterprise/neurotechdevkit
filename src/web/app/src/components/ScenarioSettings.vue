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
      <div v-if="isPreBuilt">
        <label for="scenario" class="form-label">Scenarios:</label>
        <select id="scenario" class="form-select" v-model="selectedScenario">
          <option value selected>Select a scenario</option>
          <option v-for="(value, key) in filteredBuiltInScenarios" :key="key" :value="key">
            {{ value.title }}
          </option>
        </select>
      </div>
      <div v-else>
        <form @submit.prevent>
          <div class="mb-3">
            <label class="form-label" for="loadedCTs">Available CTs</label>
            <select class="form-select" size="3" id="loadedCTs" v-model="selectedCTScan">
              <option v-for="ct in $store.getters.ctScans" :key="ct" :value="ct.filename">{{ ct.filename }}</option>
            </select>
          </div>
          <div class="mb-3" v-if="is2d">
            <label for="ctSliceAxis" class="form-label" data-bs-toggle="tooltip" data-bs-placement="right"
              title="Axis along which to slice the 3D field to be recorded">Axis</label>
            <select class="form-select" aria-label="Axis" id="ctSliceAxis" v-if="is2d" v-model="ctSliceAxis">
              <option value selected>Select an axis</option>
              <option value="x">X</option>
              <option value="y">Y</option>
              <option value="z">Z</option>
            </select>
          </div>
          <div class="mb-3" v-if="is2d">
            <label for="ctSlicePosition" class="form-label" data-bs-toggle="tooltip" data-bs-placement="right"
              title="The position (in meters) along the slice axis at which the slice of the 3D field should be made">Distance
              from Origin (m)</label>
            <input type="number" step="any" class="form-control" id="ctSlicePosition" placeholder="0.0"
              v-model="ctSlicePosition" />
          </div>
        </form>
        <div class="mb-3">
          <label for="ctFiles" class="form-label"
            title="Select the CT file and the file containing the mapping between layers and masks">CT
            and mapping
            Files</label>
          <input class="form-control" type="file" id="ctFiles" onchange="filesChosen(this)" multiple />
        </div>
        <div class="mb-3">
          <button id="fileUploadButton" class="btn btn-primary" type="button" onclick="fileUploadClicked(this)" disabled>
            Upload new CT
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      selectedCTScan: '',
      selectedScenario: '',
      ctSliceAxis: '',
      ctSlicePosition: 0.0,
    }
  },
  watch: {
    selectedScenario(newVal) {
      this.$store.dispatch('setScenario', newVal)
    },
    selectedCTScan(newVal) {
      this.$store.dispatch('setCTScan', newVal)
    },
  },
  computed: {
    filteredBuiltInScenarios() {
      if (this.is2d) {
        return this.$store.getters.builtInScenarios2d
      }
      return this.$store.getters.builtInScenarios3d
    },
    scenarioType: {
      get() {
        return this.$store.getters.isPreBuilt ? 'preBuilt' : 'ctScan'
      },
      set(newVal) {
        this.$store.dispatch('setIsPreBuilt', newVal === 'preBuilt')
      }
    },
    isPreBuilt() {
      return this.$store.getters.isPreBuilt
    },
    is2d() {
      return this.$store.getters.is2d
    },
  },
}
</script>

<style scoped></style>
