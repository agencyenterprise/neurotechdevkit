<template>
  <h1><span>Parameters</span></h1>
  <div>
    <div class="mb-3 btn-group">
      <input :disabled="is2dDisabled" class="btn-check" type="radio" id="is2d" value="2D" v-model="simulationType"
        name="simulation">
      <label :class="{ 'disabled': is2dDisabled, 'btn-smaller': true }" class="btn btn-primary" for="is2d">2D</label>
      <input :disabled="is3dDisabled" class="btn-check" type="radio" id="is3d" value="3D" v-model="simulationType"
        name="simulation">
      <label :class="{ 'disabled': is3dDisabled, 'btn-smaller': true }" class="btn btn-primary" for="is3d">3D</label>
    </div>
    <AccordionItem v-for="(item, index) in accordionItems" :key="index" :title="item.title" :index="index"
      :is-open="opened === index" @toggle="accordionToggle" :disabled="item.disabled">
      <component :is="item.component" :key="item.title" />
    </AccordionItem>
  </div>
</template>

<script>
import { mapGetters, mapActions } from 'vuex';
import { EventBus } from '../event-bus';
import ScenarioSettings from './ScenarioSettings.vue';
import DisplaySettings from './DisplaySettings.vue';
import TransducersSettings from './TransducersSettings.vue';
import TargetSettings from './TargetSettings.vue';
import SimulationSettings from './SimulationSettings.vue';
import AccordionItem from './AccordionItem.vue';

export default {
  components: {
    ScenarioSettings,
    DisplaySettings,
    TransducersSettings,
    TargetSettings,
    SimulationSettings,
    AccordionItem
  },
  data() {
    return {
      opened: null
    };
  },
  computed: {
    ...mapGetters(['is2d', 'hasSimulation', 'isRunningSimulation']),
    ...mapGetters('scenarioSettings', ['scenario', 'isScenarioValid']),
    accordionItems() {
      const items = [
        { title: 'Scenario', component: 'ScenarioSettings', disabled: false }, // Scenario is always enabled
        ...(!this.is2d ? [{ title: 'Display', component: 'DisplaySettings', disabled: !this.isScenarioValid }] : []),
        { title: 'Transducers', component: 'TransducersSettings', disabled: !this.isScenarioValid },
        { title: 'Target', component: 'TargetSettings', disabled: !this.isScenarioValid },
        { title: 'Simulation settings', component: 'SimulationSettings', disabled: !this.isScenarioValid }
      ];
      return items;
    },
    simulationType: {
      get() {
        return this.is2d ? '2D' : '3D';
      },
      set(value) {
        this.set2d(value === '2D');
      }
    },
    is3dDisabled() {
      return this.isRunningSimulation || (this.hasSimulation && this.is2d);
    },
    is2dDisabled() {
      return this.isRunningSimulation || (this.hasSimulation && !this.is2d);
    },
    backendUrl() {
      return process.env.VUE_APP_BACKEND_URL; // Or `import.meta.env.VUE_APP_BACKEND_URL` for Vue.js 3
    }
  },
  methods: {
    ...mapActions(['set2d', 'getInitialData', 'reset']),
    accordionToggle(index) {
      if (!this.accordionItems[index].disabled) {
        this.opened = this.opened === index ? null : index;
      }
    },
    resetAccordion() {
      this.opened = null; 
    },
  },
  beforeUnmount() {
    EventBus.off('reset-parameters-panel', this.resetAccordion);
  },
  mounted() {
    this.getInitialData();
    EventBus.on('reset-parameters-panel', this.resetAccordion);
  },
}
</script>

<style scoped>
.btn-primary.disabled,
.btn-primary:disabled {
  background-color: #ccc;
  border-color: #ccc;
  pointer-events: none;
  /* Prevent all interactions */
}

/* General styles */
h1 {
  font-size: 1.75rem;
  color: #333;
  margin-bottom: 1.5rem;
  text-align: center;
}

h1 span {
  padding: 0.25rem 0.5rem;
  border-radius: 0.25rem;
}

/* Button group styles */
.btn-group {
  display: flex;
  justify-content: center;
  margin-bottom: 1.5rem;
}

.btn-check {
  display: none;
  /* Hide the default radio input */
}

.btn-smaller {
  padding: 0.5rem 1rem; /* Smaller padding */
  font-size: 0.875rem; /* Smaller font size */
  margin: 0 0.25rem; /* Space out buttons */
  flex-grow: 1; /* Make buttons take up equal width */
}

.btn-primary {
  font-weight: bold;
  color: #fff;
  background-color: #007bff;
  border-color: #007bff;
  transition: background-color 0.3s ease-in-out, color 0.3s ease-in-out;
}

.btn-primary:hover,
.btn-primary:focus {
  background-color: #0056b3;
  color: #fff;
}

.btn-primary.active,
.btn-primary:active {
  background-color: #003580;
  color: #fff;
}

.btn-wide {
  padding: 0.75rem 1.5rem;
  /* Increase padding for larger buttons */
  margin: 0 0.25rem;
  /* Space out buttons */
  flex-grow: 1;
  /* Make buttons take up equal width */
}
</style>