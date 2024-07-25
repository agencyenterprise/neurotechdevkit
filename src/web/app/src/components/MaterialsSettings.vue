<template>
  <div class="accordion accordion-flush" id="materialsAccordion">
    <div class="accordion-item" v-for="(properties, material) in materialProperties" :key="material" :id="material">
      <h2 class="accordion-header" :id="'flush-heading' + material">
        <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse"
          :data-bs-target="'#flush-collapse' + material" aria-expanded="false"
          :aria-controls="'flush-collapse' + material">
          {{ material }}
        </button>
      </h2>
      <div :id="'flush-collapse' + material" class="accordion-collapse collapse"
        :aria-labelledby="'flush-heading' + material" data-bs-parent="#materialsAccordion">
        <div class="accordion-body">
          <div class="mb-3">
            <label v-tooltip="'The speed of sound (in m/s)'">VP</label>
            <input :disabled="hasSimulation || isRunningSimulation" type="number" class="form-control"
              v-model.number="properties.vp" />
          </div>
          <div class="mb-3">
            <label v-tooltip="'The mass density (in kg/m³)'">RHO</label>
            <input :disabled="hasSimulation || isRunningSimulation" type="number" class="form-control"
              v-model.number="properties.rho" />
          </div>
          <div class="mb-3">
            <label v-tooltip="'The absorption (in dB/cm)'">Alpha</label>
            <input :disabled="hasSimulation || isRunningSimulation" type="number" step="any" class="form-control"
              v-model.number="properties.alpha" />
          </div>
          <div class="mb-3">
            <label class="form-label"
              v-tooltip="'The color used when rendering this material in the scenario layout plot'">Color</label>
            <input :disabled="hasSimulation || isRunningSimulation" type="color" class="form-control form-control-color"
              v-model="properties.renderColor" v-tooltip="'Click to choose your color'" />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { mapGetters, mapState, mapMutations } from 'vuex';

export default {
  computed: {
    ...mapState('simulationSettings', ['materialProperties']),
    ...mapGetters(['hasSimulation', 'isRunningSimulation']),
  },
  methods: {
    ...mapMutations('simulationSettings', ['updateMaterialProperty']),
  },
  watch: {
    materialProperties: {
      handler(newProperties) {
        for (const material in newProperties) {
          for (const prop in newProperties[material]) {
            const value = newProperties[material][prop];
            this.updateMaterialProperty({ material, prop, value });
          }
        }
      },
      deep: true,
    },
  },
};
</script>

<style scoped></style>