<template>
  <h1>
    <span>Parameters</span>
  </h1>
  <div>
    <div class="mb-3 btn-group">
      <input class="btn-check" type="radio" id="is2d" value="2D" v-model="simulationType" name="simulation" checked>
      <label class="btn btn-primary btn-wide" for="is2d">2D</label>
      <input class="btn-check" type="radio" id="is3d" value="3D" v-model="simulationType" name="simulation">
      <label class="btn btn-primary btn-wide" for="is3d">3D</label>
    </div>
    <div class="accordion-item">
      <h2 @click="accordionToggle(0)" class="accordion-header">
        Scenario
        <font-awesome-icon :icon="getIcon(0)" />
      </h2>
      <ScenarioSettings v-if="opened === 0" ref="scenarioComponent" />
    </div>
    <div class="accordion-item">
      <h2 @click="accordionToggle(1)" class="accordion-header">
        Display
        <font-awesome-icon :icon="getIcon(1)" />
      </h2>
      <DisplaySettings v-if="opened === 1" ref="displayComponent" />
    </div>
    <div class="accordion-item">
      <h2 @click="accordionToggle(2)" class="accordion-header">
        Transducers
        <font-awesome-icon :icon="getIcon(2)" />
      </h2>
      <TransducersSettings v-if="opened === 2" ref="transducersComponent" />
    </div>
    <div class="accordion-item">
      <h2 @click="accordionToggle(3)" class="accordion-header">
        Target
        <font-awesome-icon :icon="getIcon(3)" />
      </h2>
      <TargetSettings v-if="opened === 3" ref="targetComponent" />
    </div>
    <div class="accordion-item">
      <h2 @click="accordionToggle(4)" class="accordion-header">
        Simulation Settings
        <font-awesome-icon :icon="getIcon(4)" />
      </h2>
      <SimulationSettings v-if="opened === 4" ref="simulationSettingsComponent" />
    </div>
  </div>
</template>

<script>
import ScenarioSettings from './ScenarioSettings.vue'
import DisplaySettings from './DisplaySettings.vue'
import TransducersSettings from './TransducersSettings.vue'
import TargetSettings from './TargetSettings.vue'
import SimulationSettings from './SimulationSettings.vue'

export default {
  components: {
    ScenarioSettings,
    DisplaySettings,
    TransducersSettings,
    TargetSettings,
    SimulationSettings
  },
  watch: {
    simulationType(newVal) {
      this.is2d = newVal === '2D';
    }
  },
  data() {
    return {
      simulationType: '2D',
      is2d: true,
      opened: null,
      has_simulation: false,
      is_running_simulation: false,
      configuration: {},
      built_in_scenarios: [],
      materials: [],
      material_properties: [],
      transducer_types: [],
      available_cts: [],
    }
  },
  mounted() {
    this.getInitialData()
  },
  methods: {
    /**
     * Called when the simulation type is changed. (2D or 3D)
     */
    changeSimulationType() {
      this.is2d = this.simulationType === '2D';
    },

    /**
     * Toggles the accordion item.
     * @param {number} index The index of the accordion item.
     */
    accordionToggle(index) {
      this.opened = this.opened === index ? null : index;
    },

    /**
     * Fetches the initial data from the server.
     */
    getInitialData: function () {
      fetch(`http://${process.env.VUE_APP_BACKEND_URL}/info`, {
      })
        .then(response => response.json())
        .then(data => {
          console.log("Initial data", data)
          this.has_simulation = data.has_simulation
          this.is_running_simulation = data.is_running_simulation
          this.configuration = data.configuration
          this.built_in_scenarios = data.built_in_scenarios
          this.materials = data.materials
          this.material_properties = data.material_properties
          this.transducer_types = data.transducer_types
          this.available_cts = data.available_cts
        })
        .catch(error => {
          console.error('Error:', error);
        });
    },

    /**
     * Returns the correct icon for the accordion item.
     * @param {number} index The index of the accordion item.
     */
    getIcon(index) {
      return this.opened === index ? 'chevron-up' : 'chevron-down';
    },
  }
}
</script>

<style scoped>
.accordion-item {
  margin-bottom: 10px;
}

.accordion-header {
  background-color: #f5f5f5;
  padding: 10px 15px;
  cursor: pointer;
  border-bottom: 1px solid #ddd;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.accordion-content {
  padding: 10px 15px;
}

.btn-wide {
  width: 100px;
}
</style>