<template>
  <h1>
    <span>Parameters</span>
  </h1>
  <div>
    <fieldset>
      <div class="mb-3 btn-group">
        <input type="radio" class="btn-check" name="options" id="is2d" autocomplete="off" v-model="is2d"
          @change="changeSimulationType(true)" checked />
        <label class="btn btn-primary" for="is2d">2D</label>

        <input type="radio" class="btn-check" name="options" id="is3d" autocomplete="off" v-model="is2d"
          @change="changeSimulationType(false)" />
        <label class="btn btn-primary" for="is3d">3D</label>
      </div>
    </fieldset>
    <form @submit.prevent>
      <div class="accordion" id="parameters-accordion">
        <div class="accordion-item" v-for="(item, index) in accordionItems" :key="index">
          <h2 class="accordion-header" :id="'heading' + index">
            <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse"
              :data-bs-target="'#collapse' + index" aria-expanded="false" :aria-controls="'collapse' + index">
              {{ item.title }}
            </button>
          </h2>
          <div :id="'collapse' + index" class="accordion-collapse collapse" aria-labelledby="headingOne"
            data-bs-parent="#parameters-accordion">
            <div class="accordion-body">
              <component :is="item.component" />
            </div>
          </div>
        </div>
      </div>
    </form>
  </div>
</template>

<script>
import Scenario from './Scenario.vue'
import Display from './Display.vue'
import Transducers from './Transducers.vue'
import Target from './Target.vue'
import SimulationSettings from './SimulationSettings.vue'

export default {
  name: 'HelloWorld',
  props: {
    msg: String
  },
  components: {
    Scenario,
    Display,
    Transducers,
    Target,
    SimulationSettings
  },
  data() {
    return {
      is2d: true,
      accordionItems: [
        { title: 'Scenario', component: 'Scenario' },
        { title: 'Display', component: 'Display' },
        { title: 'Transducers', component: 'Transducers' },
        { title: 'Target', component: 'Target' },
        { title: 'Simulation Settings', component: 'SimulationSettings' }
      ]
    }
  },
  methods: {
    changeSimulationType(value) {
      this.is2d = value;
    }
  }
}
</script>

<style scoped></style>