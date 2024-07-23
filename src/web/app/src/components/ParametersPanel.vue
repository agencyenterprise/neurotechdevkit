<template>
  <h1><span>Parameters</span></h1>
  <div>
    <div class="mb-3 btn-group">
      <input class="btn-check" type="radio" id="is2d" value="2D" v-model="simulationType" name="simulation">
      <label class="btn btn-primary btn-wide" for="is2d">2D</label>
      <input class="btn-check" type="radio" id="is3d" value="3D" v-model="simulationType" name="simulation">
      <label class="btn btn-primary btn-wide" for="is3d">3D</label>
    </div>
    <AccordionItem v-for="(item, index) in accordionItems" :key="index" :title="item.title" :index="index"
      :is-open="opened === index" @toggle="accordionToggle">
      <component :is="item.component" :key="item.title" />
    </AccordionItem>
  </div>
</template>

<script>
import { mapGetters, mapActions } from 'vuex';
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
    ...mapGetters({
      is2d: 'is2d'
    }),
    accordionItems() {
      const items = [
        { title: 'Scenario', component: 'ScenarioSettings' },
        // Include DisplaySettings only when !is2d (3D mode)
        ...(!this.is2d ? [{ title: 'Display', component: 'DisplaySettings' }] : []),
        { title: 'Transducers', component: 'TransducersSettings' },
        { title: 'Target', component: 'TargetSettings' },
        { title: 'Simulation settings', component: 'SimulationSettings' }
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
    backendUrl() {
      return process.env.VUE_APP_BACKEND_URL; // Or `import.meta.env.VUE_APP_BACKEND_URL` for Vue.js 3
    }
  },
  methods: {
    ...mapActions({
      set2d: 'set2d',
      setInitialValues: 'setInitialValues'
    }),
    accordionToggle(index) {
      this.opened = this.opened === index ? null : index;
    },
    getInitialData() {
      fetch(`http://${this.backendUrl}/info`)
        .then(response => {
          if (response.ok) {
            return response.json();
          }
          throw new Error('Network response was not ok.');
        })
        .then(data => {
          this.setInitialValues(data);
        })
        .catch(error => {
          console.error('Error:', error);
        });
    },
  },
  mounted() {
    this.getInitialData();
  }
}
</script>

<style scoped>
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

/* Accordion styles */
.accordion-item {
  margin-bottom: 1rem;
  border: 1px solid #e0e0e0;
  border-radius: 0.25rem;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.accordion-header {
  background-color: #f5f5f5;
  padding: 0.75rem 1rem;
  cursor: pointer;
  border-bottom: 1px solid #ddd;
  display: flex;
  justify-content: space-between;
  align-items: center;
  transition: background-color 0.3s ease-in-out;
}

.accordion-header:hover {
  background-color: #e2e6ea;
}

.accordion-content {
  padding: 1rem;
  background-color: #fff;
  border-top: none;
}

/* Ensure that the accordion content area does not have a top border */
.accordion-content:first-child {
  border-top: none;
}

/* Optional: Add a 'plus' and 'minus' icon for accordion headers */
.accordion-header::after {
  content: '\002B';
  /* Unicode plus sign */
  font-size: 1.25rem;
  color: #007bff;
}

.accordion-header[aria-expanded="true"]::after {
  content: '\2212';
  /* Unicode minus sign */
}

/* Optional: Add transition for the accordion content */
.accordion-content {
  transition: max-height 0.3s ease-in-out, padding 0.3s ease-in-out, visibility 0.3s ease-in-out;
  overflow: hidden;
  max-height: 0;
  visibility: hidden;
}

.accordion-content[aria-expanded="true"] {
  max-height: 1000px;
  /* Arbitrary large height for smooth animation */
  visibility: visible;
}
</style>