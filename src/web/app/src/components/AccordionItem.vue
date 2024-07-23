<template>
  <div class="accordion-item">
    <h2 :class="{ 'accordion-header': true, 'disabled': disabled }" @click="$emit('toggle', index)">
      {{ title }}
      <font-awesome-icon :icon="icon" />
    </h2>
    <div v-show="isOpen" class="accordion-content">
      <!-- This slot will be replaced by the content passed to the component -->
      <slot></slot>
    </div>
  </div>
</template>

<script>
import { FontAwesomeIcon } from '@fortawesome/vue-fontawesome';

export default {
  components: {
    FontAwesomeIcon
  },
  props: {
    disabled: {
      type: Boolean,
      default: false
    },
    title: {
      type: String,
      required: true
    },
    index: {
      type: Number,
      required: true
    },
    isOpen: {
      type: Boolean,
      required: true
    }
  },
  computed: {
    icon() {
      return this.isOpen ? 'chevron-up' : 'chevron-down';
    }
  }
}
</script>

<style scoped>
.accordion-item {
  margin-bottom: .5rem;
}

.accordion-header {
  padding: 10px 15px;
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  align-items: center;
  user-select: none;
}

.accordion-header.disabled {
  color: #a1a1a1;
  cursor: not-allowed;
  pointer-events: none;
}

.accordion-content {
  padding: 10px 15px;
}

.accordion-content>* {
  margin-bottom: 0;
}
</style>