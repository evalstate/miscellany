import { defineRoutesSetup } from '@slidev/types'

// Slidev v52 only enables the built-in /print route for export/build modes.
// Re-add it in dev so VS Code Simple Browser can show a live vertical slide stack.
export default defineRoutesSetup((routes) => {
  if (!routes.some(route => route.path === '/print')) {
    const notFoundIndex = routes.findIndex(route => route.name === 'NotFound')
    routes.splice(notFoundIndex >= 0 ? notFoundIndex : routes.length, 0, {
      name: 'print',
      path: '/print',
      component: () => import('@slidev/client/pages/print.vue'),
    })
  }

  return routes
})
