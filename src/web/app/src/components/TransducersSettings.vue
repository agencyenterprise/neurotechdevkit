<template>
  <div class="mb-3">
    <label for="transducerType" class="form-label">Source Transducer</label>
    <select class="form-select" id="transducerType" name="transducerType" v-model="selectedTransducer"
      @change="transducerSelected" aria-label="Select a transducer">
      <option value selected>-- Select transducer --</option>
      <option v-for="transducer in transducers" :key="transducer.value" :value="transducer.value">
        {{ transducer.text }}
      </option>
    </select>
  </div>
  <div v-if="selectedTransducer === 'PointTransducer'">
    <PointTransducer />
  </div>
  <div v-else-if="selectedTransducer === 'PhasedArrayTransducer'">
    <PhasedArrayTransducer />
  </div>
  <div v-else-if="selectedTransducer === 'FocusedTransducer'">
    <FocusedTransducer />
  </div>
  <div v-else-if="selectedTransducer === 'PlanarTransducer'">
    <PlanarTransducer />
  </div>
</template>

<script>
import PointTransducer from './transducers/PointTransducer.vue'
import PhasedArrayTransducer from './transducers/PhasedArrayTransducer.vue'
import FocusedTransducer from './transducers/FocusedTransducer.vue'
import PlanarTransducer from './transducers/PlanarTransducer.vue'

export default {
  components: {
    PointTransducer,
    PhasedArrayTransducer,
    FocusedTransducer,
    PlanarTransducer,
  },
  data() {
    return {
      selectedTransducer: '',
      transducers: [
        { text: 'Point Source', value: 'PointTransducer' },
        { text: 'Phased Array Source', value: 'PhasedArrayTransducer' },
        { text: 'Focused Source', value: 'FocusedTransducer' },
        { text: 'Planar Source', value: 'PlanarTransducer' },
      ],
    }
  },
  methods: {
    transducerSelected() {
      // handle the change event here
    },
  },
}
</script>

<style scoped></style>