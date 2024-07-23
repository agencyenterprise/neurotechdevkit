<template>
  <div v-if="!hasSimulation" class="transducers-settings">
    <div class="mb-3">
      <label class="form-label">Source transducer</label>
      <select class="form-select" v-model="selectedTransducer">
        <option value selected>-- Select transducer --</option>
        <option v-for="transducer in supportedTransducers" :key="transducer.value" :value="transducer.value">
          {{ transducer.text }}
        </option>
      </select>
    </div>
    <div v-if="selectedTransducer === 'pointSource'">
      <PointTransducer ref="transducerComponent" :readOnly="hasSimulation" />
    </div>
    <div v-else-if="selectedTransducer === 'phasedArraySource'">
      <PhasedArrayTransducer ref="transducerComponent" :readOnly="hasSimulation" />
    </div>
    <div v-else-if="selectedTransducer === 'focusedSource'">
      <FocusedTransducer ref="transducerComponent" :readOnly="hasSimulation" />
    </div>
    <div v-else-if="selectedTransducer === 'planarSource'">
      <PlanarTransducer ref="transducerComponent" :readOnly="hasSimulation" />
    </div>

    <div class="transducer-controls">
      <button class="btn btn-primary" @click="addTransducerClick" v-if="selectedTransducer && !isEditMode">
        Add Transducer
      </button>
      <button class="btn btn-secondary" @click="cancelTransducerClick" v-if="selectedTransducer">
        {{ hasSimulation ? 'Close' : 'Cancel' }}
      </button>
      <button class="btn btn-success" @click="saveTransducerClick" v-if="isEditMode && !hasSimulation">
        Save Transducer
      </button>
    </div>
  </div>

  <div v-if="transducers.length" class="card transducers-list">
    <div class="transducers-list-header">Added transducers</div>
    <ul class="list-group list-group-flush">
      <li class="list-group-item d-flex align-items-center" v-for="(transducer, index) in transducers" :key="index">
        <span class="transducer-type">{{ transducer.transducerType }}</span> <!-- Display transducer name or type -->
        <!-- Conditionally render the Edit/View and Delete buttons -->
        <div class="ms-auto button-group">
          <button class="btn btn-sm btn-outline-warning" @click="editTransducerClick(index)">
            <i class="fas fa-edit"></i> {{ hasSimulation ? 'View' : 'Edit' }}
          </button>
          <button v-if="!hasSimulation" class="btn btn-sm btn-outline-danger" @click="deleteTransducerClick(index)">
            <i class="fas fa-trash-alt"></i> Delete
          </button>
        </div>
      </li>
    </ul>
  </div>
</template>

<script>
import { mapState, mapGetters, mapActions } from 'vuex';

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
    ...mapGetters(['hasSimulation']),
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


<style scoped>
.transducers-settings{
  margin-bottom: 2rem;
}

.transducers-list {
  background-color: #f9f9f9;
  /* Light background to differentiate the list */
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  /* Add shadow for depth */
  border: none;
  /* Remove border for a cleaner look */
}

.transducers-list-header {
  padding: 0.75rem 1rem;
  background-color: #788fa8;
  color: white;
  font-weight: bold;
}

.list-group-item {
  border: none;
  padding: 0.75rem 1rem;
  border-bottom: 1px solid #e0e0e0;
  /* Add a subtle divider between items */
}

.list-group-item:last-child {
  border-bottom: none;
  /* Remove divider for the last item */
}

.transducer-type {
  font-weight: bold;
  /* Makes the transducer type text bold */
  padding-right: 1rem;
  /* Add padding to separate from buttons */
}

.button-group button {
  margin-left: 0.5rem;
  /* Adds spacing between buttons */
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  /* Consistent shadow with the list */
}

.button-group i {
  margin-right: 0.25rem;
  /* Adds spacing between icon and text */
}

.button-group button:hover {
  background-color: #f8f9fa;
  /* Light background on hover */
}

/* Style for the 'Add Transducer' button to make it stand out */
.btn-primary {
  background-color: #007bff;
  border-color: #007bff;
}

/* Style for the 'Cancel' button to make it less prominent */
.btn-secondary {
  background-color: #6c757d;
  border-color: #6c757d;
}

/* Style for the 'Save Transducer' button to indicate a positive action */
.btn-success {
  background-color: #28a745;
  border-color: #28a745;
}

/* Ensure buttons have consistent width */
.btn-sm {
  width: 90px;
}

.transducer-controls {
  display: flex;
  justify-content: center;
  /* Center the buttons */
  gap: 1rem;
  /* Add space between buttons */
  margin-top: 1rem;
  /* margin-bottom: 2rem; */
}

.transducer-controls button {
  padding: 0.5rem 1rem;
  transition: background-color 0.2s, box-shadow 0.2s;
  font-size: 1rem;
  border: none;
  border-radius: 0.25rem;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.transducer-controls .btn-primary {
  background-color: #007bff;
  color: #ffffff;
}

.transducer-controls .btn-secondary {
  background-color: #6c757d;
  color: #ffffff;
}

.transducer-controls .btn-success {
  background-color: #28a745;
  color: #ffffff;
}

.transducer-controls button:hover {
  background-color: darken(#007bff, 5%);
  /* Slightly darken the button on hover */
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
  /* Increase shadow for a "lifted" effect */
}

.transducer-controls button:active {
  box-shadow: none;
  /* Flatten button when clicked */
}

.transducer-controls button:disabled {
  background-color: #e0e0e0;
  color: #a1a1a1;
  cursor: not-allowed;
  /* Show that the button is not clickable */
}

/* Ensure buttons have consistent width */
.transducer-controls button {
  min-width: 120px;
  /* Set a minimum width for all buttons */
  text-align: center;
  /* Center the text and icons */
}
</style>