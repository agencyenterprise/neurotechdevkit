<template>
  <h1>
    <span>Parameters</span>
  </h1>
  <div>
    <div class="mb-3 btn-group">
      <input class="btn-check" type="radio" id="is2d" value="2D" v-model="simulationType" name="simulation">
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
  emits: ['new-image-generated'],
  components: {
    ScenarioSettings,
    DisplaySettings,
    TransducersSettings,
    TargetSettings,
    SimulationSettings
  },
  data: () => ({
    opened: null
  }),
  computed: {
    simulationType: {
      get() {
        return this.$store.getters.is2d ? '2D' : '3D'
      },
      set(newVal) {
        this.$store.dispatch('set2d', newVal === '2D')
      }
    }
  },
  mounted() {
    this.getInitialData()
  },
  methods: {
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
          this.$store.dispatch('setInitialValues', data)
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