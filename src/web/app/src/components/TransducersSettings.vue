<template>
  <div class="mb-3">
    <label class="form-label">Source Transducer</label>
    <select class="form-select" v-model="selectedTransducer">
      <option value selected>-- Select transducer --</option>
      <option v-for="transducer in supportedTransducers" :key="transducer.value" :value="transducer.value">
        {{ transducer.text }}
      </option>
    </select>
  </div>
  <div v-if="selectedTransducer === 'pointSource'">
    <PointTransducer ref="transducerComponent" />
  </div>
  <div v-else-if="selectedTransducer === 'phasedArraySource'">
    <PhasedArrayTransducer ref="transducerComponent" />
  </div>
  <div v-else-if="selectedTransducer === 'focusedSource'">
    <FocusedTransducer ref="transducerComponent" />
  </div>
  <div v-else-if="selectedTransducer === 'planarSource'">
    <PlanarTransducer ref="transducerComponent" />
  </div>
  <div class="transducer-controls">
    <button class="btn btn-primary" @click="addTransducerClick" v-if="selectedTransducer && !isEditMode">
      Add Transducer
    </button>
    <button class="btn btn-secondary" @click="cancelTransducerClick" v-if="selectedTransducer">
      Cancel
    </button>
    <button class="btn btn-success" @click="saveTransducerClick" v-if="isEditMode">
      Save Transducer
    </button>
  </div>

  <div class="card">
    <ul class="list-group list-group-flush">
      <li class="list-group-item d-flex" v-for="(transducer, index) in transducers" :key="index">
        {{ transducer.transducerType }} <!-- Display transducer name or type -->
        <div class="ms-auto">
          <button class="btn btn-sm btn-warning" @click="editTransducerClick(index)">Edit</button>
          <button class="btn btn-sm btn-danger" @click="deleteTransducerClick(index)">Delete</button>
        </div>
      </li>
    </ul>
  </div>
</template>

<script>
import { mapState, mapActions } from 'vuex';

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
      isEditMode: false,
      editIndex: -1,
      supportedTransducers: [
        { text: 'Point Source', value: 'pointSource' },
        { text: 'Phased Array Source', value: 'phasedArraySource' },
        { text: 'Focused Source', value: 'focusedSource' },
        { text: 'Planar Source', value: 'planarSource' },
      ],
    }
  },
  computed: {
    ...mapState('transducersSettings', ['transducers']),
  },
  methods: {
    ...mapActions('transducersSettings', ['addTransducer']),
    addTransducerClick() {
      if (this.selectedTransducer && this.$refs.transducerComponent) {
        const settings = this.$refs.transducerComponent.getTransducerSettings();
        this.addTransducer(settings);
      }
    },
    editTransducerClick(index) {
      this.isEditMode = true;
      this.editIndex = index;
      const transducerToEdit = this.transducers[index];
      this.selectedTransducer = transducerToEdit.transducerType;
      // Populate the transducer component with the selected transducer's settings
      this.$nextTick(() => {
        if (this.$refs.transducerComponent) {
          this.$refs.transducerComponent.setTransducerSettings(transducerToEdit);
        } else {
          console.error('Transducer component not found');
        }
      });
    },

    deleteTransducerClick(index) {
      this.$store.commit('transducersSettings/deleteTransducer', index);
    },

    saveTransducerClick() {
      if (this.selectedTransducer && this.$refs.transducerComponent && this.isEditMode) {
        const updatedSettings = this.$refs.transducerComponent.getTransducerSettings();
        this.$store.commit('transducersSettings/updateTransducer', {
          index: this.editIndex,
          settings: updatedSettings
        });
        this.resetEditMode();
      }
    },

    cancelTransducerClick() {
      this.resetEditMode();
    },

    resetEditMode() {
      this.isEditMode = false;
      this.editIndex = -1;
      this.selectedTransducer = ''; // Reset selection
    },
  },
}
</script>

<style scoped></style>