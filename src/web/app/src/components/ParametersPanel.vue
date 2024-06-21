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
        { title: 'Simulation Settings', component: 'SimulationSettings' }
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