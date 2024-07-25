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
          <form :ref="getFormRef(material)" @input="onInputChanged(material)">
            <div class="mb-3">
              <label v-tooltip="'The speed of sound (in m/s)'">VP</label>
              <input :disabled="hasSimulation || isRunningSimulation" type="number" class="form-control"
                v-model.number="properties.vp" required />
            </div>
            <div class="mb-3">
              <label v-tooltip="'The mass density (in kg/m³)'">RHO</label>
              <input :disabled="hasSimulation || isRunningSimulation" type="number" class="form-control"
                v-model.number="properties.rho" required />
            </div>
            <div class="mb-3">
              <label v-tooltip="'The absorption (in dB/cm)'">Alpha</label>
              <input :disabled="hasSimulation || isRunningSimulation" type="number" step="any" class="form-control"
                v-model.number="properties.alpha" required />
            </div>
            <div class="mb-3">
              <label class="form-label"
                v-tooltip="'The color used when rendering this material in the scenario layout plot'">Color</label>
              <input :disabled="hasSimulation || isRunningSimulation" type="color"
                class="form-control form-control-color" v-model="properties.renderColor"
                v-tooltip="'Click to choose your color'" required />
            </div>
          </form>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { Collapse } from 'bootstrap';
import { mapGetters, mapState, mapMutations } from 'vuex';

export default {
  computed: {
    ...mapState('simulationSettings', ['materialProperties']),
    ...mapGetters(['hasSimulation', 'isRunningSimulation']),
  },
  methods: {
    ...mapMutations('simulationSettings', ['updateMaterialProperty']),
    getFormRef(material) {
      return material
    },
    validate() {
      let allValid = true;
      for (const material in this.materialProperties) {
        const form = this.$refs[material][0];
        if (form && !form.checkValidity()) {
          allValid = false;
          this.openAccordion(material);
          form.reportValidity();
          break; // Stop checking after the first invalid form
        }
      }
      return allValid;
    },
    onInputChanged(material) {
      const form = this.$refs[material][0];
      if (form && !form.checkValidity()) {
        form.reportValidity();
      }
    },
    openAccordion(material) {
      const accordionItem = this.$refs[material][0].closest('.accordion-item');
      const collapseElementId = accordionItem.querySelector('.accordion-collapse').id;
      const collapseElement = document.getElementById(collapseElementId);
      const bsCollapse = new Collapse(collapseElement, { toggle: false });
      bsCollapse.show();
    },
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